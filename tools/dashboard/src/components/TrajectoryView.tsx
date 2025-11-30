import React, { useMemo } from 'react';
import { Stage, Layer, Line, Circle, Arrow } from 'react-konva';
import { useSimulationStore } from '../store/simulationStore';

interface TrajectoryViewProps {
  width: number;
  height: number;
}

export const TrajectoryView: React.FC<TrajectoryViewProps> = ({ width, height }) => {
  const { data, getCurrentPoint } = useSimulationStore();
  const currentPoint = getCurrentPoint();

  // Calculate scaling and offset to fit trajectory in view
  const { scale, offsetX, offsetY } = useMemo(() => {
    if (!data || data.steps.length === 0) return { scale: 1, offsetX: 0, offsetY: 0 };

    const xCoords = data.steps.map((p) => p.x);
    const yCoords = data.steps.map((p) => p.y);

    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);

    const dataWidth = maxX - minX;
    const dataHeight = maxY - minY;

    // Add some padding (10%)
    const padding = Math.max(dataWidth, dataHeight) * 0.1;

    const scaleX = width / (dataWidth + padding * 2);
    const scaleY = height / (dataHeight + padding * 2);
    const scale = Math.min(scaleX, scaleY);

    // Center the trajectory
    // Note: Y-axis is flipped in canvas (positive down), but usually simulation is positive up (or down depending on coord system)
    // Let's assume standard math coords (Y up) and flip it for canvas

    const offsetX = width / 2 - (minX + dataWidth / 2) * scale;
    const offsetY = height / 2 + (minY + dataHeight / 2) * scale; // Flip Y

    return { scale, offsetX, offsetY };
  }, [data, width, height]);

  const points = useMemo(() => {
    if (!data) return [];
    return data.steps.flatMap((p) => [
      p.x * scale + offsetX,
      -p.y * scale + offsetY, // Flip Y
    ]);
  }, [data, scale, offsetX, offsetY]);

  if (!data) return <div className="flex items-center justify-center h-full">No Data</div>;

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <Stage width={width} height={height - 40}>
        <Layer>
          {/* Grid or Axis could go here */}

          {/* Full Trajectory */}
          <Line points={points} stroke="#cbd5e1" strokeWidth={2} tension={0} />

          {/* Current Position Marker */}
          {currentPoint && (
            <>
              {/* Vehicle Position */}
              <Circle
                x={currentPoint.x * scale + offsetX}
                y={-currentPoint.y * scale + offsetY}
                radius={5}
                fill="#3b82f6"
              />

              {/* Heading Arrow */}
              <Arrow
                x={currentPoint.x * scale + offsetX}
                y={-currentPoint.y * scale + offsetY}
                points={[0, 0, 20, 0]}
                rotation={(-currentPoint.yaw * 180) / Math.PI} // Konva uses degrees, and Y flip affects rotation direction
                pointerLength={10}
                pointerWidth={10}
                fill="#ef4444"
                stroke="#ef4444"
                strokeWidth={2}
              />
            </>
          )}
        </Layer>
      </Stage>
    </div>
  );
};
