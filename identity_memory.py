import cv2
import numpy as np


class IdentityMemory:

    def __init__(self):

        self.next_id = 1

        self.memory = {}

        self.raw_ids = {}

        self.max_lost_frames = 1800

        self.threshold = 0.72

    def similarity(self, feature1, feature2):

        distance = cv2.compareHist(
            feature1.astype("float32"),
            feature2.astype("float32"),
            cv2.HISTCMP_BHATTACHARYYA
        )

        return max(0, 1 - distance)

    def create_identity(self, features, frame_number, raw_id):

        identity = self.next_id

        self.next_id += 1

        self.memory[identity] = {

            "features": features.copy(),

            "last_seen": frame_number

        }

        if raw_id is not None:
            self.raw_ids[raw_id] = (
                identity, frame_number
            )

        return identity

    def assign_identity(
        self,
        raw_id,
        features,
        frame_number
    ):

        # Check whether the tracker still recognizes this track.

        if raw_id is not None and raw_id in self.raw_ids:

            identity, last_frame = self.raw_ids[raw_id]

            if frame_number - last_frame <= 30:

                if identity in self.memory:

                    record = self.memory[identity]

                    record["features"] = (
                        0.85 * record["features"]
                        + 0.15 * features
                    )

                    record["last_seen"] = frame_number

                    self.raw_ids[raw_id] = (
                        identity, frame_number
                    )

                    return identity

        # Search previously lost identities.

        candidates = []

        for identity, record in self.memory.items():

            lost_time = (
                frame_number - record["last_seen"]
            )

            if 5 < lost_time <= self.max_lost_frames:

                score = self.similarity(
                    features,
                    record["features"]
                )

                candidates.append((score, identity))

        candidates.sort(reverse=True)

        if candidates:

            best_score, best_identity = candidates[0]

            second_score = (
                candidates[1][0]
                if len(candidates) > 1
                else 0
            )

            if (
                best_score >= self.threshold
                and best_score - second_score >= 0.08
            ):

                record = self.memory[best_identity]

                record["features"] = (
                    0.85 * record["features"]
                    + 0.15 * features
                )

                record["last_seen"] = frame_number

                if raw_id is not None:
                    self.raw_ids[raw_id] = (
                        best_identity, frame_number
                    )

                return best_identity

        # If no reliable match exists, create a new identity.

        return self.create_identity(
            features,
            frame_number,
            raw_id
        )

    def cleanup(self, frame_number):

        expired = []

        for identity, record in self.memory.items():

            if (
                frame_number - record["last_seen"]
                > self.max_lost_frames
            ):

                expired.append(identity)

        for identity in expired:
            del self.memory[identity]

        self.raw_ids = {

            raw_id: (identity, last_frame)

            for raw_id, (identity, last_frame)
            in self.raw_ids.items()

            if identity in self.memory
            and frame_number - last_frame <= 30

        }
