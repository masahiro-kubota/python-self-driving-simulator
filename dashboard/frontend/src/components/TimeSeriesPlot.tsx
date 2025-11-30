import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Paper, Typography, Box, useTheme } from '@mui/material';
import { useSimulationStore } from '../store/simulationStore';
import type { TrajectoryPoint } from '../types';

interface TimeSeriesPlotProps {
  title: string;
  dataKey: string;
  color?: string;
  unit?: string;
  height?: number;
}

export const TimeSeriesPlot: React.FC<TimeSeriesPlotProps> = ({
  title,
  dataKey,
  color,
  unit = '',
  height = 200,
}) => {
  const theme = useTheme();
  const lineColor = color || theme.palette.primary.main;
  const { data, currentTime } = useSimulationStore();
  const currentPoint = useSimulationStore((state) => state.getCurrentPoint());

  // Prepare plot data
  const plotData = useMemo((): Plotly.Data[] => {
    if (!data) return [];
    return [
      {
        x: data.steps.map((p) => p.timestamp),
        y: data.steps.map((p) => p[dataKey as keyof TrajectoryPoint] as number),
        mode: 'lines',
        type: 'scatter',
        name: title,
        line: {
          color: lineColor,
          width: 2,
        },
        hovertemplate: 'Time: %{x:.2f}s<br>Value: %{y:.3f} ' + unit + '<extra></extra>',
      },
    ];
  }, [data, dataKey, title, lineColor, unit]);

  // Layout configuration
  const layout = useMemo((): Partial<Plotly.Layout> => {
    if (!data || data.steps.length === 0) {
      return {
        height,
        margin: { l: 50, r: 30, t: 10, b: 40 },
        paper_bgcolor: theme.palette.background.paper,
        plot_bgcolor: theme.palette.background.default,
      };
    }

    const timestamps = data.steps.map((p) => p.timestamp);
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);

    return {
      height,
      margin: { l: 50, r: 30, t: 10, b: 40 },
      xaxis: {
        title: { text: 'Time (s)' },
        range: [minTime, maxTime],
        gridcolor: theme.palette.divider,
        zerolinecolor: theme.palette.divider,
      },
      yaxis: {
        title: { text: unit },
        gridcolor: theme.palette.divider,
        zerolinecolor: theme.palette.divider,
      },
      hovermode: 'closest',
      dragmode: 'pan',
      paper_bgcolor: theme.palette.background.paper,
      plot_bgcolor: theme.palette.background.default,
      font: {
        color: theme.palette.text.primary,
      },
      shapes: [
        {
          type: 'line',
          x0: currentTime,
          x1: currentTime,
          y0: 0,
          y1: 1,
          yref: 'paper',
          line: {
            color: theme.palette.error.main,
            width: 2,
            dash: 'dash',
          },
        },
      ],
    };
  }, [data, currentTime, height, unit, theme]);

  if (!data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography>No Data</Typography>
      </Box>
    );
  }

  const currentValue = currentPoint ? (currentPoint[dataKey as keyof TrajectoryPoint] ?? 0) : 0;

  const config: Partial<Plotly.Config> = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
    scrollZoom: true,
  };

  return (
    <Paper elevation={2} sx={{ overflow: 'hidden', borderRadius: 2 }}>
      <Box
        sx={{
          p: 1.5,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Typography variant="subtitle1" fontWeight="bold" color="primary">
          {title}
        </Typography>
        <Typography variant="h6" sx={{ fontFamily: 'monospace', color: lineColor }}>
          {(typeof currentValue === 'number' ? currentValue : 0).toFixed(3)} {unit}
        </Typography>
      </Box>
      <Box sx={{ bgcolor: 'background.default' }}>
        <Plot data={plotData} layout={layout} config={config} style={{ width: '100%' }} />
      </Box>
    </Paper>
  );
};
