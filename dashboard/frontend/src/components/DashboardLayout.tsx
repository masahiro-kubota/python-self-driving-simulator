import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Paper,
  Box,
  Stack,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import { TimeSlider } from './TimeSlider';
import { TrajectoryView } from './TrajectoryView';
import { TimeSeriesPlot } from './TimeSeriesPlot';
import { useSimulationStore } from '../store/simulationStore';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

export const DashboardLayout: React.FC = () => {
  const data = useSimulationStore((state) => state.data);

  if (!data)
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <Typography>Loading data...</Typography>
      </Box>
    );

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <AppBar
          position="static"
          color="transparent"
          elevation={0}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
              Simulation Dashboard
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {data.metadata.controller} | {data.metadata.execution_time}
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
          <Stack spacing={3}>
            <TimeSlider />

            <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', lg: 'row' } }}>
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Paper sx={{ p: 2, height: '600px', display: 'flex', flexDirection: 'column' }}>
                  <Box sx={{ flexGrow: 1, bgcolor: '#000', borderRadius: 1, overflow: 'hidden' }}>
                    <TrajectoryView width={600} height={600} />
                  </Box>
                </Paper>
              </Box>

              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Stack spacing={2} sx={{ width: '100%', minWidth: 0 }}>
                  <TimeSeriesPlot title="Velocity" dataKey="velocity" unit="m/s" height={130} />
                  <TimeSeriesPlot title="Steering" dataKey="steering" unit="rad" height={130} />
                  <TimeSeriesPlot
                    title="Acceleration"
                    dataKey="acceleration"
                    unit="m/s²"
                    height={130}
                  />
                  <TimeSeriesPlot title="Yaw" dataKey="yaw" unit="rad" height={150} />
                </Stack>
              </Box>
            </Box>
          </Stack>
        </Container>
      </Box>
    </ThemeProvider>
  );
};
