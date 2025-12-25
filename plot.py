import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple
import spiceypy

@dataclass
class VisibilitySegment:
    """Class to store visibility segment data"""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    elevation_times: List[pd.Timestamp]
    elevation_values: List[float]

class VisibilityPlotter:
    def __init__(self, start_date_str: str, end_date_str: str, min_elevation: float, colormap: str, min_gap_duration: float):
        self.start_date = pd.to_datetime(start_date_str)
        self.end_date = pd.to_datetime(end_date_str)
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.min_elevation = min_elevation
        self.colormap = plt.get_cmap(colormap)
        self.min_gap_duration = min_gap_duration
        self.figure = None
        self.ax_elevation = None
        self.ax_gantt = None
        
        # Data storage
        self.station_segments: Dict[str, List[VisibilitySegment]] = {}
        self.station_colors: Dict[str, str] = {}
        self.dishes_df = None
        self.coverage_gaps = []
    
    def _format_station_name(self, station_name: str) -> str:
        """Format station name to match dishes.csv.
        
        Args:
            station_name: Raw station name (e.g. 'New_York_NY' or 'DSS-14')
            
        Returns:
            Formatted name (e.g. 'New York, NY' or 'DSS-14 (G)')
        """
        if station_name.startswith('DSS-'):
            # Extract dish number
            dish_num = int(station_name[4:])
            # Determine complex based on dish number
            if 10 <= dish_num <= 29:
                complex = 'G'
            elif 30 <= dish_num <= 49:
                complex = 'C'
            elif 50 <= dish_num <= 69:
                complex = 'M'
            else:
                complex = '?'
            return f"{station_name} ({complex})"
        else:
            # Replace underscores with spaces and add comma before state
            parts = station_name.split('_')
            if len(parts) >= 2:
                # Join all parts except the last with spaces, then add comma before the last part
                return f"{' '.join(parts[:-1])}, {parts[-1]}"
            return station_name

    def _get_sorted_station_files(self, calculated_dir: Path) -> List[Path]:
        """Get station files sorted by DSS complex (C, M, G) and longitude."""
        
        def get_complex_priority(station_name: str) -> int:
            """Get sorting priority for DSS complexes (C=0, M=1, G=2, non-DSS=3)"""
            if not station_name.startswith('DSS-'):
                return 3
            
            dish_num = int(station_name[4:])
            if 30 <= dish_num <= 49:  # Complex C
                return 0
            elif 50 <= dish_num <= 69:  # Complex M
                return 1
            elif 10 <= dish_num <= 29:  # Complex G
                return 2
            return 3

        def sort_key(file_path: Path) -> Tuple[int, int]:
            station_name = file_path.stem
            complex_priority = get_complex_priority(station_name)
            
            if station_name.startswith('DSS-'):
                # For DSS stations, use dish number as secondary key
                secondary_key = int(station_name[4:])
            else:
                # For other stations, use negative longitude as secondary key
                secondary_key = int(-self.dishes_df.loc[self._format_station_name(station_name), 'Longitude (deg)'])
            
            return (complex_priority, secondary_key)
        
        return sorted(calculated_dir.iterdir(), key=sort_key)
    
    def load_data(self):
        """Load and process all station data into visibility segments."""
        calculated_dir = Path(__file__).parent / 'calculated'
        self.dishes_df = pd.read_csv(Path(__file__).parent / 'data/dishes.csv', index_col='Location')
        
        for csv_file in self._get_sorted_station_files(calculated_dir):
            station_name = csv_file.stem
            df = pd.read_csv(csv_file)
            df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
            
            # Filter data for time range and minimum elevation
            df_filtered = df[
                (df['Time (UTC)'] >= self.start_date) & 
                (df['Time (UTC)'] <= self.end_date) &
                (df['Elevation (deg)'] >= self.min_elevation)
            ]
            
            if len(df_filtered) > 0:
                segments = self._process_station_data(df_filtered)
                if segments:
                    self.station_segments[station_name] = segments
    
    def _process_station_data(self, df: pd.DataFrame) -> List[VisibilitySegment]:
        """Process a single station's data into visibility segments."""
        segments = []
        df = df.reset_index(drop=True)  # Reset index to use positional indexing
        
        # Find continuous visibility periods
        start_idx = None
        prev_time = None
        
        for i in range(len(df)):
            time = df['Time (UTC)'].iloc[i]
            
            # Check for time gap
            if prev_time is not None and (time - prev_time) > pd.Timedelta(minutes=30):
                if start_idx is not None:
                    # End the current segment at the last point
                    segments.append(self._create_segment(df, start_idx, i))
                    start_idx = i  # Start new segment at current point
            
            # Normal segment processing
            if start_idx is None:
                start_idx = i
            
            prev_time = time
        
        # Add last segment
        if start_idx is not None:
            segments.append(self._create_segment(df, start_idx, len(df)))
        
        return segments
    
    def _create_segment(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> VisibilitySegment:
        """Create a visibility segment from a data frame slice."""
        segment_df = df.iloc[start_idx:end_idx]
        return VisibilitySegment(
            start_time=segment_df['Time (UTC)'].iloc[0],
            end_time=segment_df['Time (UTC)'].iloc[-1],
            elevation_times=segment_df['Time (UTC)'].tolist(),
            elevation_values=segment_df['Elevation (deg)'].tolist()
        )
    
    def create_plots(self):
        """Create both visibility and GANTT plots in a single figure."""
        # Create figure with extra space at bottom
        self.figure = plt.figure(figsize=(10, 8))  # Made taller
        
        # Create gridspec with extra space at bottom and more space on left
        gs = self.figure.add_gridspec(
            2, 1, 
            height_ratios=[1, 1], 
            hspace=0.3,    # Increased spacing between plots
        )
        
        # Create subplots using gridspec
        self.ax_elevation = self.figure.add_subplot(gs[0])
        self.ax_gantt = self.figure.add_subplot(gs[1])
        
        # Hide x-axis for elevation plot since we're sharing
        self.ax_elevation.xaxis.set_visible(False)
        
        # Create a new axis for longitude, sharing x with others
        self.ax_longitude = self.ax_gantt.twiny()
        self.ax_longitude.spines['top'].set_position(('outward', 25))  # Reduced from 40 to 25
        
        # Remove padding on x-axis for all plots
        self.ax_elevation.margins(x=0)
        self.ax_gantt.margins(x=0)
        self.ax_longitude.margins(x=0)
        
        # Find coverage gaps before creating plots
        self._find_coverage_gaps()
        
        # Create the main plots
        self._create_elevation_plot()
        self._create_gantt_plot()
        self._create_longitude_axis()
        
        # Plot the gaps last so we can use the final plot layout
        self._plot_coverage_gaps()
        
        # Adjust layout to prevent overlapping
        self.figure.align_ylabels()
    
    def _create_elevation_plot(self):
        """Create the elevation angle plot."""
        plt.sca(self.ax_elevation)
        
        for i, (station_name, segments) in enumerate(self.station_segments.items()):
            # Combine all segments for continuous plotting with gaps
            times = []
            elevations = []
            
            for segment in segments:
                if times:  # Add nan between segments to create visual break
                    times.append(segment.elevation_times[0])
                    elevations.append(np.nan)
                times.extend(segment.elevation_times)
                elevations.extend(segment.elevation_values)
            
            color = self.colormap(i / len(self.station_segments))
            # Use formatted station name in legend
            line = self.ax_elevation.plot(times, elevations, 
                                        label=self._format_station_name(station_name), 
                                        color=color)[0]
            self.station_colors[station_name] = color
        
        self._setup_elevation_formatting()
    
    def _find_coverage_gaps(self):
        """Find periods where no station has visibility."""
        # Combine all segments from all stations and sort by start time
        all_segments = []
        for segments in self.station_segments.values():
            all_segments.extend(segments)
        all_segments.sort(key=lambda x: x.start_time)
        
        # Find gaps between segments
        current_time = self.start_date
        self.coverage_gaps = []
        
        for segment in all_segments:
            # Check if this segment overlaps with current time
            if segment.start_time > current_time:
                # Found a gap
                self.coverage_gaps.append(VisibilitySegment(
                    start_time=current_time,
                    end_time=segment.start_time,
                    elevation_times=[],
                    elevation_values=[]
                ))
            # Update current time if this segment extends beyond it
            current_time = max(current_time, segment.end_time)
        
        # Check for final gap
        if current_time < self.end_date:
            self.coverage_gaps.append(VisibilitySegment(
                start_time=current_time,
                end_time=self.end_date,
                elevation_times=[],
                elevation_values=[]
            ))

    def _plot_coverage_gaps(self):
        """Plot coverage gaps as vertical shaded areas across both plots."""
        if not self.coverage_gaps:
            return
        
        # Create a new axes that spans the entire figure and shares the x-axis
        gap_ax = self.figure.add_axes(
            self.ax_elevation.get_position(),
            frameon=False,
            sharex=self.ax_elevation
        )
        gap_ax.set_ylim(0, 1)  # Use unit height
        
        # Hide all spines, ticks, and labels
        for spine in gap_ax.spines.values():
            spine.set_visible(False)
        gap_ax.set_xticks([])
        gap_ax.set_yticks([])
        
        # Extend gap axis to cover the space between plots
        pos = gap_ax.get_position()
        gap_ax.set_position([
            pos.x0,
            self.ax_gantt.get_position().y0,  # Bottom of GANTT
            pos.width,
            self.ax_elevation.get_position().y1 - self.ax_gantt.get_position().y0  # Full height
        ])
        
        for gap in self.coverage_gaps:
            # Add duration label at the bottom of the gantt plot
            gap_duration = gap.end_time - gap.start_time
            minutes = gap_duration.total_seconds() / 60
            
            # Only plot gaps longer than minimum duration
            if minutes > self.min_gap_duration:
                # Draw the gap shading using data coordinates
                gap_ax.axvspan(
                    gap.start_time, gap.end_time,
                    color='red', alpha=0.1, zorder=0
                )

                duration_str = f"{minutes:.0f}min"
                self.ax_gantt.text(
                    gap.start_time + (gap.end_time - gap.start_time) / 2,
                    self.ax_gantt.get_ylim()[0] - 0.5,
                    duration_str,
                    verticalalignment='top',
                    horizontalalignment='center',
                    fontsize=8,
                    fontfamily='monospace',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                    zorder=3
                )

    def _create_gantt_plot(self):
        """Create the GANTT chart."""
        plt.sca(self.ax_gantt)
        stations = list(self.station_segments.keys())
        
        # Skip labels near edge of plot
        boundary_threshold = pd.Timedelta(minutes=1)
        
        # Plot station segments
        for i, station_name in enumerate(stations):
            segments = self.station_segments[station_name]
            for segment in segments:
                # Draw the horizontal line for the segment
                self.ax_gantt.hlines(
                    i, segment.start_time, segment.end_time,
                    linewidth=10, color=self.station_colors[station_name],
                    zorder=2  # Ensure segments appear above gap shading
                )
                
                # Add start time label, shifting to plot boundary if needed
                if segment.start_time - self.start_date > boundary_threshold:
                    start_time_str = segment.start_time.strftime('%H:%M')
                    
                    # Create text object to measure its width
                    text = self.ax_gantt.text(
                        segment.start_time, i - 0.15, start_time_str,
                        verticalalignment='top',
                        horizontalalignment='right',
                        fontsize=8,
                        fontfamily='monospace',
                        zorder=3
                    )
                    
                    # Get the width of the text in data coordinates
                    fig = plt.gcf()
                    renderer = fig.canvas.get_renderer()
                    bbox = text.get_window_extent(renderer)
                    bbox_data = bbox.transformed(self.ax_gantt.transData.inverted())
                    text_width = bbox_data.width
                    # add 10% padding to the text width
                    text_width = text_width * 1.1
                    
                    # Convert text width to timedelta
                    time_range = self.end_date - self.start_date
                    ax_width = self.ax_gantt.get_xlim()[1] - self.ax_gantt.get_xlim()[0]
                    width_timedelta = pd.Timedelta(seconds=text_width * time_range.total_seconds() / ax_width)
                    
                    # If text would extend before plot start, adjust position
                    if segment.start_time - width_timedelta < self.start_date:
                        text.set_x(self.start_date + width_timedelta)
                    
                # Add end time label if not too close to right boundary
                if self.end_date - segment.end_time > boundary_threshold:
                    end_time_str = segment.end_time.strftime('%H:%M')
                    self.ax_gantt.text(
                        segment.end_time, i - 0.15, end_time_str,
                        verticalalignment='top',
                        horizontalalignment='left',
                        fontsize=8,
                        fontfamily='monospace',
                        zorder=3  # Ensure labels appear above everything
                    )
        
        self._setup_gantt_formatting(stations)
    
    def _create_longitude_axis(self):
        """Create the longitude axis showing mean solar time."""
        # Get the time limits from the gantt plot
        time_min, time_max = self.ax_gantt.get_xlim()
        
        # Convert datetime numbers to actual datetime objects
        time_min_dt = mdates.num2date(time_min)
        time_max_dt = mdates.num2date(time_max)
        
        # Set the longitude limits (-180 to 180)
        self.ax_longitude.set_xlim(time_min, time_max)
        
        # Get Earth's radii to compute flattening coefficient
        _, radii = spiceypy.bodvrd('EARTH', 'RADII', 3)
        re = radii[0]  # Equatorial radius
        rp = radii[2]  # Polar radius
        f = (re - rp) / re  # Flattening coefficient
        
        # Get the time ticks from the gantt plot
        time_ticks = self.ax_gantt.get_xticks()
        time_tick_dates = [mdates.num2date(t) for t in time_ticks]
        # Convert time ticks to longitudes
        longitude_labels = []
        for dt in time_tick_dates:
            et = spiceypy.datetime2et(dt)
            spoint, _trgepc, _srfvec = spiceypy.subpnt(
                method='INTERCEPT/ELLIPSOID', 
                target='EARTH', 
                et=et, 
                fixref='IAU_EARTH', 
                abcorr='CN+S', 
                obsrvr='-242' # LTB
            )
            
            # Convert rectangular coordinates to planetographic coordinates
            lon, lat, alt = spiceypy.recpgr('EARTH', spoint, re, f)
            # Convert radians to degrees
            lon_deg = lon * 180 / np.pi
            
            # Format the longitude label
            if abs(lon_deg) > 180:  # cycle to -180
                label = f"{lon_deg - 360:.0f}°E"
            else:
                label = f"{lon_deg:.0f}°E"
            longitude_labels.append(label)
        
        self.ax_longitude.set_xticks(time_ticks)
        self.ax_longitude.set_xticklabels(longitude_labels)
        
        # Set longitude axis label
        self.ax_longitude.set_xlabel('Sub-LTB Longitude')
        
        # Adjust appearance
        self.ax_longitude.grid(False)
        for spine in ['top', 'right']:
            self.ax_longitude.spines[spine].set_visible(True)
        for spine in ['bottom', 'left']:
            self.ax_longitude.spines[spine].set_visible(False)
    
    def _setup_elevation_formatting(self):
        """Setup elevation plot formatting and labels."""
        self.ax_elevation.set_ylabel('Elevation (deg)', labelpad=-50)  
        
        self.ax_elevation.set_title(
            f'LTB Visibility (>{self.min_elevation}°) from {self.start_date_str} to {self.end_date_str}'
        )
        
        self.ax_elevation.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_elevation.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        self.ax_elevation.set_ylim(bottom=self.min_elevation)
        self.ax_elevation.grid(True)
        self.ax_elevation.legend()
    
    def _setup_gantt_formatting(self, stations: List[str]):
        """Setup GANTT chart formatting and labels."""
        self.ax_gantt.set_yticks(range(len(stations)))
        # Use formatted station names for y-axis labels
        self.ax_gantt.set_yticklabels([self._format_station_name(station) for station in stations])
        self.ax_gantt.set_xlabel('Time (UTC)')
        self.ax_gantt.invert_yaxis()  # Make order match the elevation plot legend
        
        # Configure time axis
        self.ax_gantt.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        self.ax_gantt.tick_params(axis='x', which='major', pad=8)  # Add padding below tick labels
        
        self.ax_gantt.grid(True, axis='x')
        self.ax_gantt.spines['top'].set_visible(False)
        self.ax_gantt.spines['right'].set_visible(False)
        
        # Add padding at the bottom for time labels
        self.ax_gantt.set_ylim(bottom=len(stations)-0.5, top=-0.5)
    
    def save_plot(self, filename: str):
        """Save the combined plot to a file."""
        if self.figure:
            self.figure.savefig(filename, bbox_inches='tight')
            plt.close('all')

def main():
    parent_dir = Path(__file__).parent
    data_dir = parent_dir / 'data'

    parser = argparse.ArgumentParser(
        description='Generate LTB visibility plots for a given time range.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--bsp', default=str(data_dir / 'ltb_trj_od006v1.bsp'), help='Path to LTB trajectory kernel file, such as ltb_trj_od005v1.bsp')
    parser.add_argument('--start', type=str, default='2025-03-20T00:00:00',
                       help='Start date and time in ISO format (YYYY-MM-DDThh:mm:ss). UTC. Defaults to today at 00:00 UTC')
    parser.add_argument('--end', type=str, default='2025-03-22T00:00:00',
                       help='End date and time in ISO format (YYYY-MM-DDThh:mm:ss). UTC. Defaults to start + 2 days')
    parser.add_argument('--min-elevation', type=float, default=10.0,
                       help='Minimum elevation angle in degrees for visibility')
    parser.add_argument('--colormap', type=str, default='jet',
                       help='Matplotlib colormap to use for station colors')
    parser.add_argument('--output', type=str, default='visibility.pdf',
                       help='Output filename for the combined plot')
    parser.add_argument('--min-gap-duration', type=float, default=0.0,
                       help='Minimum gap duration in minutes to display')
    args = parser.parse_args()
    
    # Load required SPICE kernels
    spiceypy.furnsh(str(args.bsp))
    spiceypy.furnsh(str(data_dir / 'naif0012.tls'))
    spiceypy.furnsh(str(data_dir / 'pck00010.tpc'))
    spiceypy.furnsh(str(data_dir / 'earth_latest_high_prec.bpc'))
    
    plotter = VisibilityPlotter(args.start, args.end, args.min_elevation, args.colormap, args.min_gap_duration)
    plotter.load_data()
    plotter.create_plots()
    plotter.save_plot(args.output)

if __name__ == "__main__":
    main()
