export interface TrajectoryPoint {
  timestamp: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  velocity: number;
  acceleration: number;
  steering: number;
}

export interface SimulationMetadata {
  controller: string;
  simulator: string;
  execution_time: string;
  [key: string]: string | number;
}

export interface MapLine {
  points: { x: number; y: number }[];
}

export interface MapPolygon {
  points: { x: number; y: number }[];
}

export interface VehicleParams {
  width: number;
  length: number;
  wheelbase: number;
  front_overhang: number;
  rear_overhang: number;
}

export interface ObstacleShape {
  type: 'rectangle' | 'circle';
  width?: number | null;
  length?: number | null;
  radius?: number | null;
}

export interface ObstaclePosition {
  x: number;
  y: number;
  yaw: number;
}

export interface TrajectoryWaypoint {
  time: number;
  x: number;
  y: number;
  yaw: number;
}

export interface ObstacleTrajectory {
  type: 'waypoint';
  interpolation: 'linear' | 'cubic_spline';
  waypoints: TrajectoryWaypoint[];
  loop: boolean;
}

export interface Obstacle {
  type: 'static' | 'dynamic';
  shape: ObstacleShape;
  position?: ObstaclePosition | null;
  trajectory?: ObstacleTrajectory | null;
}

export interface SimulationData {
  metadata: SimulationMetadata;
  vehicle_params?: VehicleParams;
  steps: TrajectoryPoint[];
  map_lines?: MapLine[];
  map_polygons?: MapPolygon[];
  obstacles?: Obstacle[];
}

declare global {
  interface Window {
    SIMULATION_DATA: SimulationData;
  }
}
