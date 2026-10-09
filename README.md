# Gazelle
Appearance-based gaze estimation from face images, trained on MPIIFaceGaze.

# Overview

_Gazelle_ predicts where a person is looking from a single RGB face image. Given a cropped face, the model regresses a 3D gaze direction, expressed as pitch and yaw angles, without any calibration or special hardware.

The project is trained and evaluated on the MPIIFaceGaze dataset using a leave-one-person-out protocol.
