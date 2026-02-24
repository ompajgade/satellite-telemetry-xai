FAULT_RULES = {
    "Magnetometer": {
        "Sensor_Drift": 0.4,
        "Electromagnetic_Interference": 0.3,
        "Calibration_Error": 0.3
    },
    "Sun_Sensor": {
        "Sun_Blinding": 0.5,
        "Sensor_Saturation": 0.3,
        "Thermal_Noise": 0.2
    },
    "ADCS": {
        "Attitude_Estimation_Error": 0.6,
        "Power_Instability": 0.4
    }
}