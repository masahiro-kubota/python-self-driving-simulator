
from core.utils.mcap_utils import read_messages
import sys

mcap_path = "scripts/system_identification/data/rosbag2_autoware_0.mcap"
topics_to_check = [
    "/control/command/control_cmd",
    "/vehicle/status/steering_status",
    "/localization/kinematic_state",
    "/sensing/gnss/pose",
    "/sensing/gnss/pose_with_covariance"
]

print(f"Checking topics in {mcap_path}...")
counts = {t: 0 for t in topics_to_check}
sample_msgs = {}

for topic, msg, t in read_messages(mcap_path, topics_to_check):
    if topic in counts:
        counts[topic] += 1
        if topic not in sample_msgs:
            sample_msgs[topic] = msg

print("\nTopic Counts:")
for t, count in counts.items():
    print(f"  {t}: {count}")

print("\nSample Message Structures (Attribute names):")
for t, msg in sample_msgs.items():
    print(f"\nTopic: {t}")
    if hasattr(msg, "__dict__"):
        print(f"  Attributes: {list(msg.__dict__.keys())}")
        # Try to drill down a bit
        if t == "/sensing/gnss/pose":
             if hasattr(msg, "pose"):
                 print(f"  msg.pose type: {type(msg.pose)}")
                 if hasattr(msg.pose, "pose"):
                      print(f"  msg.pose.pose type: {type(msg.pose.pose)}")
                      if hasattr(msg.pose.pose, "position"):
                           print(f"  msg.pose.pose.position.z: {getattr(msg.pose.pose.position, 'z', 'N/A')}")
                 elif hasattr(msg.pose, "position"):
                      print(f"  msg.pose.position.z: {getattr(msg.pose.position, 'z', 'N/A')}")
