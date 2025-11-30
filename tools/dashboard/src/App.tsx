import { useEffect } from 'react';
import { DashboardLayout } from './components/DashboardLayout';
import { useSimulationStore } from './store/simulationStore';

function App() {
  const setData = useSimulationStore((state) => state.setData);

  useEffect(() => {
    // Check for injected data
    if (window.SIMULATION_DATA) {
      setData(window.SIMULATION_DATA);
    } else {
      // Fallback for development (fetch from local file or mock)
      console.warn("No injected data found. Trying to fetch 'simulation_log.json'...");
      fetch('/simulation_log.json')
        .then((res) => res.json())
        .then((data) => setData(data))
        .catch((err) => console.error('Failed to load simulation data:', err));
    }
  }, [setData]);

  return <DashboardLayout />;
}

export default App;
