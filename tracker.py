from identity_memory import IdentityMemory


class IdentityTracker:

    def __init__(self):

        self.memory = IdentityMemory()

        self.object_ids = {}
        self.next_object_id = 1
        self.previous_centers = {}

    def update(self, detections, frame_number):

        tracked_objects = []

        for detection in detections:

            raw_id = detection["raw_id"]
            class_id = detection["class_id"]
            class_name = detection["class_name"]

            if class_id == 0:

                identity = self.memory.assign_identity(
                    raw_id,
                    detection["features"],
                    frame_number
                )

                identity_label = f"Person {identity}"

            else:

                key = (class_id, raw_id)

                if raw_id is None:
                    key = (class_id, "unknown", self.next_object_id)
                    self.next_object_id += 1

                if key not in self.object_ids:
                    self.object_ids[key] = self.next_object_id
                    self.next_object_id += 1

                identity = self.object_ids[key]
                identity_label = f"{class_name} {identity}"

            x1, y1, x2, y2 = detection["bbox"]

            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            track_key = (class_id, identity)

            previous = self.previous_centers.get(track_key)

            if previous is None:
                movement = "Unknown"
                speed = 0.0
            else:
                dx = center[0] - previous[0]
                dy = center[1] - previous[1]

                speed = (dx * dx + dy * dy) ** 0.5
                movement = "Moving" if speed > 2 else "Stationary"

            self.previous_centers[track_key] = center

            tracked_objects.append({
                "identity": identity,
                "identity_label": identity_label,
                "class_id": class_id,
                "class_name": class_name,
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "movement": movement,
                "speed": speed
            })

        self.memory.cleanup(frame_number)

        return tracked_objects
