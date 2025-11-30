import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useSimulationStore } from '../store/simulationStore';

interface TimeSeriesPlotProps {
  title: string;
  dataKey: string;
  color: string;
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
  const { data, currentTime } = useSimulationStore();
  const currentPoint = useSimulationStore((state) => state.getCurrentPoint());

  if (!data)
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography>No Data</Typography>
      </Box>
    );

  const currentValue = currentPoint
    ? ((currentPoint[dataKey as keyof typeof currentPoint] as number) ?? 0)
    : 0;

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
        <Typography variant="h6" sx={{ fontFamily: 'monospace', color: color }}>
          {(typeof currentValue === 'number' ? currentValue : 0).toFixed(3)} {unit}
        </Typography>
      </Box>
      <Box sx={{ height: height, bgcolor: 'background.default' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.steps} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis
              dataKey="timestamp"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(val) => val.toFixed(1)}
              stroke="#999"
            />
            <YAxis stroke="#999" />
            <Tooltip
              labelFormatter={(label) => `Time: ${Number(label).toFixed(2)}s`}
              formatter={(value: number) => [`${value.toFixed(3)} ${unit}`, title]}
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
            />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
            <ReferenceLine x={currentTime} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
};
