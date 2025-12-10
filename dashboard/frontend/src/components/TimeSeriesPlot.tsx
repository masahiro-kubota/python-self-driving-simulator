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

  // Prepare static plot data (only updates when data changes)
  const staticTrace = useMemo((): Plotly.Data | null => {
    if (!data) return null;

    const yValues = data.steps.map((p) => p[dataKey as keyof TrajectoryPoint] as number);

    return {
      x: data.steps.map((p) => p.timestamp),
      y: yValues,
      mode: 'lines',
      type: 'scatter',
      name: title,
      line: {
        color: lineColor,
        width: 2,
      },
      hovertemplate: 'Time: %{x:.2f}s<br>Value: %{y:.3f} ' + unit + '<extra></extra>',
    };
  }, [data, dataKey, title, lineColor, unit]);

  // Calculate Y range for cursor (memoized with staticTrace)
  const yRange = useMemo(() => {
    if (!staticTrace || !data) return { min: 0, max: 1 };
    // Recalculate or store from staticTrace?
    // Accessing yValues from data again is cheap.
    const yValues = data.steps.map((p) => p[dataKey as keyof TrajectoryPoint] as number);
    const min = Math.min(...yValues);
    const max = Math.max(...yValues);
    const padding = (max - min) * 0.1 || 1.0; // Handle flat line case
    return { min: min - padding, max: max + padding };
  }, [data, dataKey, staticTrace]);

  // Prepare dynamic plot data
  const plotData = useMemo((): Plotly.Data[] => {
    if (!staticTrace) return [];

    const cursorTrace: Plotly.Data = {
      x: [currentTime, currentTime],
      y: [yRange.min, yRange.max],
      mode: 'lines',
      type: 'scatter',
      name: 'Current Time',
      line: {
        color: theme.palette.error.main,
        width: 2,
        dash: 'dash',
      },
      hoverinfo: 'skip',
      showlegend: false,
    };

    return [staticTrace, cursorTrace];
  }, [staticTrace, currentTime, yRange, theme]);

  // Layout configuration
  const layout = useMemo((): Partial<Plotly.Layout> => {
    if (!data || data.steps.length === 0) {
      return {
        // height, // Let container control height
        margin: { l: 50, r: 30, t: 10, b: 40 },
        paper_bgcolor: theme.palette.background.paper,
        plot_bgcolor: theme.palette.background.default,
      };
    }

    const timestamps = data.steps.map((p) => p.timestamp);
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);

    return {
      // height, // Let container control height
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
      uirevision: data.metadata.execution_time || 'default',
    };
  }, [data, unit, theme]); // Removed height form dependency

  const currentValue = currentPoint ? (currentPoint[dataKey as keyof TrajectoryPoint] ?? 0) : 0;

  const config = useMemo(
    (): Partial<Plotly.Config> => ({
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d'],
      scrollZoom: true,
      responsive: true,
    }),
    []
  );

  if (!data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography>No Data</Typography>
      </Box>
    );
  }

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
      {/* Set explicit height for the plot container */}
      <Box sx={{ bgcolor: 'background.default', height: height, position: 'relative' }}>
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </Box>
    </Paper>
  );
};
