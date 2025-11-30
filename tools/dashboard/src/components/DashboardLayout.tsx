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
    primary: { main: '#90caf9' },
    secondary: { main: '#f48fb1' },
    background: { default: '#0a1929', paper: '#132f4c' },
  },
});

export const DashboardLayout: React.FC = () => {
  const { data } = useSimulationStore();

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
              <Box sx={{ flex: 1 }}>
                <Paper sx={{ p: 2, height: '600px', display: 'flex', flexDirection: 'column' }}>
                  <Box sx={{ flexGrow: 1, bgcolor: '#000', borderRadius: 1, overflow: 'hidden' }}>
                    <TrajectoryView width={600} height={600} />
                  </Box>
                </Paper>
              </Box>

              <Box sx={{ flex: 1 }}>
                <Stack spacing={2}>
                  <TimeSeriesPlot
                    title="Velocity"
                    dataKey="velocity"
                    color="#90caf9"
                    unit="m/s"
                    height={130}
                  />
                  <TimeSeriesPlot
                    title="Steering"
                    dataKey="steering"
                    color="#f48fb1"
                    unit="rad"
                    height={130}
                  />
                  <TimeSeriesPlot
                    title="Acceleration"
                    dataKey="acceleration"
                    color="#a5d6a7"
                    unit="m/s²"
                    height={130}
                  />
                  <TimeSeriesPlot
                    title="Yaw"
                    dataKey="yaw"
                    color="#ce93d8"
                    unit="rad"
                    height={130}
                  />
                </Stack>
              </Box>
            </Box>
          </Stack>
        </Container>
      </Box>
    </ThemeProvider>
  );
};
