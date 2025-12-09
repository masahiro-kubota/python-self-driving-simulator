import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useSimulationStore } from '../store/simulationStore';
import { useTheme } from '@mui/material';

interface TrajectoryViewProps {
  width: number;
  height: number;
}

export const TrajectoryView: React.FC<TrajectoryViewProps> = ({ width, height }) => {
  const theme = useTheme();
  const { data, getCurrentPoint } = useSimulationStore();
  const currentPoint = getCurrentPoint();

  // Prepare trajectory data
  const trajectoryData = useMemo(() => {
    if (!data || data.steps.length === 0) return [];

    const traces: Plotly.Data[] = [];

    // Map Polygons (Lanelets - drivable area)
    // Fill with background color to cover the blue outside area
    if (data.map_polygons) {
      data.map_polygons.forEach((poly, idx) => {
        traces.push({
          x: poly.points.map((p) => p.x),
          y: poly.points.map((p) => p.y),
          mode: 'lines',
          fill: 'toself',
          fillcolor: theme.palette.background.default, // Cover blue outside area
          type: 'scatter',
          name: `Lanelet ${idx + 1}`,
          line: {
            color: 'transparent',
            width: 0,
          },
          showlegend: false,
          hoverinfo: 'skip',
        });
      });
    }

    // Map lines (Lanelet)
    if (data.map_lines) {
      data.map_lines.forEach((line, idx) => {
        traces.push({
          x: line.points.map((p) => p.x),
          y: line.points.map((p) => p.y),
          mode: 'lines',
          type: 'scatter',
          name: `Map Line ${idx + 1}`,
          line: {
            color: theme.palette.grey[400],
            width: 1,
          },
          showlegend: false,
          hoverinfo: 'skip',
        });
      });
    }

    // Full trajectory path
    traces.push({
      x: data.steps.map((p) => p.x),
      y: data.steps.map((p) => p.y),
      mode: 'lines',
      type: 'scatter',
      name: 'Trajectory',
      line: {
        color: theme.palette.grey[300],
        width: 2,
      },
      showlegend: false,
      hovertemplate: 'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
    });

    // Current vehicle as rectangle
    if (currentPoint && data.vehicle_params) {
      const { width, wheelbase, front_overhang, rear_overhang } = data.vehicle_params;
      const { x, y, yaw } = currentPoint;

      // Vehicle coordinate (x, y) is at the rear axle center
      // Calculate the 4 corners of the vehicle rectangle
      // Local coordinates (vehicle frame):
      //   - rear end: -rear_overhang (behind rear axle)
      //   - front end: wheelbase + front_overhang (ahead of rear axle)
      //   - left side: -width/2
      //   - right side: width/2

      const cos_yaw = Math.cos(yaw);
      const sin_yaw = Math.sin(yaw);

      // Helper function to transform local coordinates to global
      const transform = (local_x: number, local_y: number) => ({
        x: x + local_x * cos_yaw - local_y * sin_yaw,
        y: y + local_x * sin_yaw + local_y * cos_yaw,
      });

      // Define 4 corners in local frame (rear axle center is origin)
      const rear_left = transform(-rear_overhang, width / 2);
      const rear_right = transform(-rear_overhang, -width / 2);
      const front_right = transform(wheelbase + front_overhang, -width / 2);
      const front_left = transform(wheelbase + front_overhang, width / 2);

      // Close the polygon
      const corners = [rear_left, rear_right, front_right, front_left, rear_left];

      traces.push({
        x: corners.map((c) => c.x),
        y: corners.map((c) => c.y),
        mode: 'lines',
        fill: 'toself',
        type: 'scatter',
        name: 'Vehicle',
        fillcolor: theme.palette.primary.main,
        opacity: 0.6,
        line: {
          color: theme.palette.primary.dark,
          width: 2,
        },
        showlegend: false,
        hovertemplate:
          'Vehicle\u003cbr\u003eX: %{x:.2f}\u003cbr\u003eY: %{y:.2f}\u003cbr\u003eYaw: ' +
          ((yaw * 180) / Math.PI).toFixed(1) +
          '°\u003cextra\u003e\u003c/extra\u003e',
      });
    } else if (currentPoint) {
      // Fallback to point marker if vehicle params not available
      traces.push({
        x: [currentPoint.x],
        y: [currentPoint.y],
        mode: 'markers',
        type: 'scatter',
        name: 'Current Position',
        marker: {
          color: theme.palette.primary.main,
          size: 10,
          symbol: 'circle',
        },
        showlegend: false,
        hovertemplate:
          'X: %{x:.2f}\u003cbr\u003eY: %{y:.2f}\u003cbr\u003eYaw: ' +
          ((currentPoint.yaw * 180) / Math.PI).toFixed(1) +
          '°\u003cextra\u003e\u003c/extra\u003e',
      });
    }

    return traces;
  }, [data, currentPoint, theme]);

  // Calculate layout with proper aspect ratio
  const layout = useMemo((): Partial<Plotly.Layout> => {
    if (!data || data.steps.length === 0) {
      return {
        width,
        height: height - 40,
        xaxis: { title: { text: 'X (m)' } },
        yaxis: { title: { text: 'Y (m)' }, scaleanchor: 'x', scaleratio: 1 },
        margin: { l: 50, r: 50, t: 20, b: 50 },
        paper_bgcolor: theme.palette.background.paper,
        plot_bgcolor: theme.palette.background.default,
      };
    }

    const xCoords = data.steps.map((p) => p.x);
    const yCoords = data.steps.map((p) => p.y);

    // Include map_lines in bounds calculation
    if (data.map_lines) {
      data.map_lines.forEach((line) => {
        line.points.forEach((point) => {
          xCoords.push(point.x);
          yCoords.push(point.y);
        });
      });
    }

    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);

    // Add padding
    const padding = Math.max(maxX - minX, maxY - minY) * 0.1;

    return {
      width,
      height: height - 40,
      xaxis: {
        title: { text: 'X (m)' },
        range: [minX - padding, maxX + padding],
        gridcolor: theme.palette.divider,
        zerolinecolor: theme.palette.divider,
      },
      yaxis: {
        title: { text: 'Y (m)' },
        range: [minY - padding, maxY + padding],
        scaleanchor: 'x',
        scaleratio: 1,
        gridcolor: theme.palette.divider,
        zerolinecolor: theme.palette.divider,
      },
      margin: { l: 50, r: 50, t: 20, b: 50 },
      hovermode: 'closest',
      dragmode: 'pan',
      paper_bgcolor: theme.palette.background.paper,
      plot_bgcolor:
        theme.palette.mode === 'dark'
          ? `${theme.palette.info.dark}33` // Add 20% opacity (33 in hex)
          : `${theme.palette.info.light}4D`, // Add 30% opacity (4D in hex) for outside area
      font: {
        color: theme.palette.text.primary,
      },
      annotations: currentPoint
        ? [
            {
              x: currentPoint.x + 20 * Math.cos(currentPoint.yaw),
              y: currentPoint.y + 20 * Math.sin(currentPoint.yaw),
              ax: currentPoint.x,
              ay: currentPoint.y,
              xref: 'x',
              yref: 'y',
              axref: 'x',
              ayref: 'y',
              showarrow: true,
              arrowhead: 2,
              arrowsize: 1,
              arrowwidth: 2,
              arrowcolor: theme.palette.error.main,
            },
          ]
        : [],
    };
  }, [data, currentPoint, width, height, theme]);

  const config: Partial<Plotly.Config> = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
    scrollZoom: true,
  };

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full bg-white rounded-lg shadow-md">
        No Data
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <Plot data={trajectoryData} layout={layout} config={config} />
    </div>
  );
};
