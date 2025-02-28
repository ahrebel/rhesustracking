# src/fuse_landmarks.py
def fuse_results(dlc_result, sleap_result, dlc_weight=0.9, sleap_weight=0.8):
    """
    Fuses the eye coordinates from DLC and SLEAP.
    
    If both results are available, uses weighted averaging.
    If only one is available, returns that result.
    
    Returns:
      dict: {
          'eye_coord': fused (x, y),
          'roll_angle': roll angle (from DLC if available),
          'landmarks': {
              'DLC': <DLC landmarks>,
              'SLEAP': <SLEAP landmarks>
          }
      }
    """
    if dlc_result is None and sleap_result is None:
        return None
    elif dlc_result is None:
        return sleap_result
    elif sleap_result is None:
        return dlc_result

    eye_dlc = dlc_result["eye_coord"]
    eye_sleap = sleap_result["eye_coord"]
    
    fused_eye_coord = (
        (eye_dlc[0] * dlc_weight + eye_sleap[0] * sleap_weight) / (dlc_weight + sleap_weight),
        (eye_dlc[1] * dlc_weight + eye_sleap[1] * sleap_weight) / (dlc_weight + sleap_weight)
    )
    
    # Use DLC's roll angle if available
    roll_angle = dlc_result.get("roll_angle", None)
    
    return {
        "eye_coord": fused_eye_coord,
        "roll_angle": roll_angle,
        "landmarks": {
            "DLC": dlc_result["landmarks"],
            "SLEAP": sleap_result["landmarks"]
        }
    }
