import React, { useEffect, useRef, useCallback } from 'react';
import { PlayArrow, Pause, SkipPrevious } from '@mui/icons-material';
import {
  Box,
  IconButton,
  Slider,
  Typography,
  Select,
  MenuItem,
  FormControl,
  Paper,
  Stack,
} from '@mui/material';
import { useSimulationStore } from '../store/simulationStore';

export const TimeSlider: React.FC = () => {
  const {
    currentTime,
    duration,
    isPlaying,
    togglePlay,
    setCurrentTime,
    playbackSpeed,
    setPlaybackSpeed,
  } = useSimulationStore();

  const requestRef = useRef<number | undefined>(undefined);
  const previousTimeRef = useRef<number | undefined>(undefined);

  const animate = useCallback(
    (time: number) => {
      if (previousTimeRef.current !== undefined) {
        const deltaTime = (time - previousTimeRef.current) / 1000;

        if (isPlaying) {
          let nextTime = useSimulationStore.getState().currentTime + deltaTime * playbackSpeed;
          if (nextTime > duration) {
            nextTime = duration;
            useSimulationStore.getState().togglePlay();
          }
          setCurrentTime(nextTime);
        }
      }
      previousTimeRef.current = time;
      requestRef.current = requestAnimationFrame(animate);
    },
    [isPlaying, playbackSpeed, duration, setCurrentTime]
  );

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [animate]);

  const formatTime = (time: number) => {
    return time.toFixed(2);
  };

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: 2 }}>
      <Stack direction="row" spacing={2} alignItems="center" mb={1}>
        <IconButton
          onClick={togglePlay}
          color="primary"
          sx={{ bgcolor: 'primary.main', color: 'white', '&:hover': { bgcolor: 'primary.dark' } }}
        >
          {isPlaying ? <Pause /> : <PlayArrow />}
        </IconButton>

        <IconButton onClick={() => setCurrentTime(0)}>
          <SkipPrevious />
        </IconButton>

        <Box sx={{ flexGrow: 1, mx: 2 }}>
          <Slider
            value={currentTime}
            min={0}
            max={duration}
            step={0.01}
            onChange={(_, value) => setCurrentTime(value as number)}
            valueLabelDisplay="auto"
            valueLabelFormat={formatTime}
          />
        </Box>

        <Typography
          variant="h6"
          sx={{ fontFamily: 'monospace', fontWeight: 'bold', minWidth: '80px', textAlign: 'right' }}
        >
          {formatTime(currentTime)}s
        </Typography>
      </Stack>

      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Box display="flex" alignItems="center">
          <Typography variant="body2" color="text.secondary" mr={1}>
            Speed:
          </Typography>
          <FormControl size="small" variant="standard">
            <Select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              disableUnderline
              sx={{ fontSize: '0.875rem' }}
            >
              <MenuItem value={0.5}>0.5x</MenuItem>
              <MenuItem value={1}>1.0x</MenuItem>
              <MenuItem value={2}>2.0x</MenuItem>
              <MenuItem value={5}>5.0x</MenuItem>
              <MenuItem value={10}>10.0x</MenuItem>
            </Select>
          </FormControl>
        </Box>
        <Typography variant="body2" color="text.secondary">
          Duration: {formatTime(duration)}s
        </Typography>
      </Stack>
    </Paper>
  );
};
