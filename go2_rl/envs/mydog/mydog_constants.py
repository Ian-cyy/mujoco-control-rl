"""Defines constants for the self-built dog."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "mydog"
XML = ROOT_PATH / "xmls" / "mydog_simple.xml"

ROOT_BODY = "base"

UPVECTOR_SENSOR = "upvector"
FORWARDVECTOR_SENSOR = "forwardvector"
GYRO_SENSOR = "gyro"
ACCELEROMETER_SENSOR = "accelerometer"
