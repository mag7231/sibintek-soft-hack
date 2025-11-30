# web/dashboard.py

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from datetime import timedelta

# add project root
sys.path.append(str(Path(__file__).parent.parent))

from src.database import DatabaseManager
from src.database import DatabaseManager, VideoSession


st.set_page_config(layout="wide")


def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@st.cache_resource
def get_db(db_path: str = 'data/app.db') -> DatabaseManager:
    return DatabaseManager(db_path)


def create_worker_stats_chart(worker_stats: dict) -> go.Figure:
    if not worker_stats:
        return None

    classes = list(worker_stats.keys())
    work_times = [worker_stats[cls]['work_time'] / 60 for cls in classes]
    idle_times = [worker_stats[cls]['idle_time'] / 60 for cls in classes]

    fig = go.Figure(data=[
        go.Bar(name='Active Time (Work Zone)', x=classes, y=work_times),
        go.Bar(name='Idle Time (Other Zones)', x=classes, y=idle_times)
    ])

    fig.update_layout(
        title='Worker Activity by Class',
        xaxis_title='Worker Class',
        yaxis_title='Time (minutes)',
        barmode='group',
        height=380
    )

    return fig


def create_zone_distribution_chart(workers: list) -> go.Figure:
    zone_times = {}
    for worker in workers:
        try:
            zones = json.loads(worker.zones_visited)
        except Exception:
            zones = {}
        for zone, t in zones.items():
            zone_times[zone] = zone_times.get(zone, 0) + t

    if not zone_times:
        return None

    fig = go.Figure(data=[go.Pie(labels=list(zone_times.keys()), values=list(zone_times.values()), hole=0.3)])
    fig.update_layout(title='Time Distribution Across Zones', height=380)
    return fig


def create_worker_timeline(workers: list) -> go.Figure:
    if not workers:
        return None

    df_data = []
    for worker in workers:
        df_data.append({
            'Worker': f"{worker.worker_class.capitalize()} (ID: {worker.track_id})",
            'Start': worker.first_seen,
            'End': worker.last_seen,
            'Class': worker.worker_class
        })

    df = pd.DataFrame(df_data)
    color_map = {
        'mechanic': 'lightblue',
        'worker': 'gray',
        'cleaner': 'lightgreen',
        'driver': 'pink',
        'unknown': 'lightgray'
    }

    fig = px.timeline(df, x_start='Start', x_end='End', y='Worker', color='Class', color_discrete_map=color_map)
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=420)
    return fig


def filter_workers_by_frame_range(workers, start_frame, end_frame):
    """Возвращает список WorkerActivity, у которых есть пересечение с диапазоном кадров"""
    out = []
    for w in workers:
        if w.first_frame is None or w.last_frame is None:
            continue
        if w.last_frame < start_frame or w.first_frame > end_frame:
            continue
        out.append(w)
    return out


def main():
    st.title("Railway Worker Monitoring — Dashboard")

    db = get_db()

    # sessions
    # sessions = db.session.query(db.session.mapper_registry.mapped[0].__class__).all()
    # The above is a fallback; better is to query VideoSession directly
    all_sessions = db.session.query(VideoSession).all()

    if not all_sessions:
        st.error("No video sessions found in database. Process a video first.")
        return

    session_options = {f"{s.id} — {Path(s.video_path).name} ({s.created_at.strftime('%Y-%m-%d %H:%M')})": s.id for s in all_sessions}
    selected_label = st.sidebar.selectbox("Select session", list(session_options.keys()))
    session_id = session_options[selected_label]

    # fetch session data
    stats = db.get_session_statistics(session_id)
    if not stats:
        st.error("No stats for this session")
        return

    session = stats['session']
    trains = stats['trains']
    workers = stats['workers']
    worker_stats = stats['worker_stats']
    attentions = stats['attentions']

    # header metrics
    st.header("Session info")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("**Video**")
        st.write(Path(session.video_path).name)
    with col2:
        if session.start_time and session.end_time:
            duration = session.end_time - session.start_time
        else:
            duration = timedelta(0)
        st.write("**Duration**")
        st.write(format_timedelta(duration))
    with col3:
        st.write("**FPS**")
        st.write(f"{session.fps:.1f}")
    with col4:
        st.write("**Total frames**")
        st.write(f"{session.total_frames}")

    st.write('---')

    # Frame range slider
    st.sidebar.header("Frame range for fragment")
    max_frame = session.total_frames or 0
    start_frame, end_frame = st.sidebar.slider("Select frame interval", 0, max_frame, (0, max_frame), step=max(1, max_frame//100 if max_frame>0 else 1))

    st.subheader(f"Showing statistics for frames {start_frame} — {end_frame}")

    # Filter workers by frame range
    workers_in_range = filter_workers_by_frame_range(workers, start_frame, end_frame)

    # Aggregate per-class stats for selected range
    agg = {}
    for w in workers_in_range:
        cls = w.worker_class
        overlap_start = max(w.first_frame, start_frame)
        overlap_end = min(w.last_frame, end_frame)
        overlap_frames = max(0, overlap_end - overlap_start + 1)
        overlap_seconds = overlap_frames / session.fps

        # estimate time in work zone proportionally
        try:
            zones = json.loads(w.zones_visited or '{}')
        except Exception:
            zones = {}

        work_zones = db._get_work_zones_for_class(cls)
        time_in_work = 0.0
        time_in_other = 0.0
        total_zone_time = sum(zones.values()) if zones else 0.0
        if total_zone_time > 0:
            # proportional split
            work_zone_time_total = sum(v for k, v in zones.items() if k in work_zones)
            time_in_work = overlap_seconds * (work_zone_time_total / total_zone_time)
            time_in_other = overlap_seconds - time_in_work
        else:
            time_in_work = 0.0
            time_in_other = overlap_seconds

        if cls not in agg:
            agg[cls] = {'count': 0, 'total_time': 0.0, 'work_time': 0.0, 'idle_time': 0.0}
        agg[cls]['count'] += 1
        agg[cls]['total_time'] += overlap_seconds
        agg[cls]['work_time'] += time_in_work
        agg[cls]['idle_time'] += time_in_other

    # Show metrics and charts
    st.header("Worker statistics for fragment")
    if agg:
        col1, col2, col3, col4 = st.columns(4)
        total_workers = sum(v['count'] for v in agg.values())
        with col1:
            st.metric("Workers (unique tracks)", total_workers)
        with col2:
            st.metric("Total time (min)", f"{sum(v['total_time'] for v in agg.values())/60:.1f}")
        with col3:
            st.metric("Active time (min)", f"{sum(v['work_time'] for v in agg.values())/60:.1f}")
        with col4:
            st.metric("Idle time (min)", f"{sum(v['idle_time'] for v in agg.values())/60:.1f}")

        fig = create_worker_stats_chart(agg)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        fig2 = create_zone_distribution_chart(workers_in_range)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = create_worker_timeline(workers_in_range)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("No worker activity in this frame range")

    st.write('---')

    # Trains in range
    st.header("Train events in fragment")
    trains_in_range = []
    for tr in trains:
        a = tr.arrival_frame or 0
        d = tr.departure_frame or max_frame
        if d < start_frame or a > end_frame:
            continue
        trains_in_range.append(tr)

    if trains_in_range:
        for tr in trains_in_range:
            st.write(f"Train {tr.train_number or 'unknown'} — arrival: {tr.arrival_time} (frame {tr.arrival_frame}) — departure: {tr.departure_time}")
    else:
        st.info("No trains in the selected fragment")

    st.write('---')

    # Attention events
    st.header("Attention events in fragment")
    att_in_range = [a for a in attentions if a.frame_number >= start_frame and a.frame_number <= end_frame]
    if att_in_range:
        for ev in att_in_range:
            sev_icon = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(ev.severity, '⚪')
            with st.expander(f"{sev_icon} {ev.event_type} — frame {ev.frame_number}"):
                st.write(ev.description)
                st.write(f"Track: {ev.track_id} | Class: {ev.worker_class} | Zone: {ev.zone_name}")
                cols = st.columns(3)
                if cols[0].button('Mark OK', key=f'ok_{ev.id}'):
                    db.resolve_attention_event(ev.id, 'ok')
                    st.experimental_rerun()
                if cols[1].button('No uniform', key=f'nouniform_{ev.id}'):
                    db.resolve_attention_event(ev.id, 'no_uniform')
                    st.experimental_rerun()
                if cols[2].button('Save crop', key=f'save_{ev.id}'):
                    st.info('Saving crop to S3 is not configured in this demo')
    else:
        st.info('No attention events in this fragment')

    # Export CSV
    if st.button('Export worker summary CSV'):
        rows = []
        for w in workers_in_range:
            rows.append({
                'track_id': w.track_id,
                'class': w.worker_class,
                'first_frame': w.first_frame,
                'last_frame': w.last_frame,
                'total_time_s': w.total_time
            })
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        st.download_button('Download CSV', data=csv, file_name=f'session_{session_id}_fragment_{start_frame}_{end_frame}.csv')

    db.close()


if __name__ == '__main__':
    main()
