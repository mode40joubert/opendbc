import copy
import numpy as np
from opendbc.car import CanBusBase
from opendbc.car.crc import CRC16_XMODEM
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.lead_data_ext import CanFdLeadData


class CanBus(CanBusBase):
  def __init__(self, CP, fingerprint=None, lka_steering=None) -> None:
    super().__init__(CP, fingerprint)

    if lka_steering is None:
      lka_steering = CP.flags & HyundaiFlags.CANFD_LKA_STEERING.value if CP is not None else False

    # On the CAN-FD platforms, the LKAS camera is on both A-CAN and E-CAN. LKA steering cars
    # have a different harness than the LFA steering variants in order to split
    # a different bus, since the steering is done by different ECUs.
    self._a, self._e = 1, 0
    if lka_steering:
      self._a, self._e = 0, 1

    self._a += self.offset
    self._e += self.offset
    self._cam = 2 + self.offset

  @property
  def ECAN(self):
    return self._e

  @property
  def ACAN(self):
    return self._a

  @property
  def CAM(self):
    return self._cam


def create_steering_messages(packer, CP, CAN, enabled, lat_active, apply_torque, lkas_icon):
  common_values = {
    "LKA_MODE": 2,
    "LKA_ICON": lkas_icon,
    "TORQUE_REQUEST": apply_torque,
    "LKA_ASSIST": 0,
    "STEER_REQ": 1 if lat_active else 0,
    "STEER_MODE": 0,
    "HAS_LANE_SAFETY": 0,  # hide LKAS settings
    "NEW_SIGNAL_2": 0,
    "DAMP_FACTOR": 100,  # can potentially tuned for better perf [3, 200]
  }

  lkas_values = copy.copy(common_values)
  lkas_values["LKA_AVAILABLE"] = 0

  lfa_values = copy.copy(common_values)
  lfa_values["NEW_SIGNAL_1"] = 0

  ret = []
  if CP.flags & HyundaiFlags.CANFD_LKA_STEERING:
    lkas_msg = "LKAS_ALT" if CP.flags & HyundaiFlags.CANFD_LKA_STEERING_ALT else "LKAS"
    if CP.openpilotLongitudinalControl:
      ret.append(packer.make_can_msg("LFA", CAN.ECAN, lfa_values))
    ret.append(packer.make_can_msg(lkas_msg, CAN.ACAN, lkas_values))
  else:
    ret.append(packer.make_can_msg("LFA", CAN.ECAN, lfa_values))

  return ret


def create_suppress_lfa(packer, CAN, lfa_block_msg, lka_steering_alt):
  suppress_msg = "CAM_0x362" if lka_steering_alt else "CAM_0x2a4"
  msg_bytes = 32 if lka_steering_alt else 24

  values = {f"BYTE{i}": lfa_block_msg[f"BYTE{i}"] for i in range(3, msg_bytes) if i != 7}
  values["COUNTER"] = lfa_block_msg["COUNTER"]
  values["SET_ME_0"] = 0
  values["SET_ME_0_2"] = 0
  values["LEFT_LANE_LINE"] = 0
  values["RIGHT_LANE_LINE"] = 0
  return packer.make_can_msg(suppress_msg, CAN.ACAN, values)


def create_buttons(packer, CP, CAN, cnt, btn):
  values = {
    "COUNTER": cnt,
    "SET_ME_1": 1,
    "CRUISE_BUTTONS": btn,
  }

  bus = CAN.ECAN if CP.flags & HyundaiFlags.CANFD_LKA_STEERING else CAN.CAM
  return packer.make_can_msg("CRUISE_BUTTONS", bus, values)


def create_acc_cancel(packer, CP, CAN, cruise_info_copy):
  # CAN FD camera-based SCC requires additional signals to be preserved
  # verbatim from the previous SCC_CONTROL frame to avoid checksum or
  # state validation faults. Classic CAN SCC only validates a subset.
  if CP.flags & HyundaiFlags.CANFD_CAMERA_SCC.value:
    values = {s: cruise_info_copy[s] for s in [
      "COUNTER",
      "CHECKSUM",
      "NEW_SIGNAL_1",
      "MainMode_ACC",
      "ACCMode",
      "ZEROS_9",
      "CRUISE_STANDSTILL",
      "ZEROS_5",
      "DISTANCE_SETTING",
      "VSetDis",
    ]}
  else:
    values = {s: cruise_info_copy[s] for s in [
      "COUNTER",
      "CHECKSUM",
      "ACCMode",
      "VSetDis",
      "CRUISE_STANDSTILL",
    ]}
  values.update({
    "ACCMode": 4,
    "aReqRaw": 0.0,
    "aReqValue": 0.0,
  })
  return packer.make_can_msg("SCC_CONTROL", CAN.ECAN, values)


def create_lfahda_cluster(packer, CAN, enabled, lfa_icon):
  values = {
    "HDA_ICON": 1 if enabled else 0,
    "LFA_ICON": lfa_icon,
  }
  return packer.make_can_msg("LFAHDA_CLUSTER", CAN.ECAN, values)


def create_acc_control(packer, CAN, enabled, accel_last, accel, stopping, gas_override, set_speed, hud_control,
                       lead_data: CanFdLeadData, main_cruise_enabled, tuning):
  jerk = 5
  jn = jerk / 50
  if not enabled or gas_override:
    a_val, a_raw = 0, 0
  else:
    a_raw = accel  # noqa: F841
    a_val = np.clip(accel, accel_last - jn, accel_last + jn)  # noqa: F841

  values = {
    "ACCMode": 0 if not enabled else (2 if gas_override else 1),
    "MainMode_ACC": 1 if main_cruise_enabled else 0,
    "StopReq": 1 if tuning.stopping else 0,
    "aReqValue": tuning.actual_accel,
    "aReqRaw": tuning.actual_accel,
    "VSetDis": set_speed,
    "JerkLowerLimit": tuning.jerk_lower,
    "JerkUpperLimit": tuning.jerk_upper,

    "ACC_ObjDist": int(lead_data.lead_distance),
    "ACC_ObjRelSpd": lead_data.lead_rel_speed,
    "ObjValid": int(not lead_data.lead_visible),
    "SCC_ObjSta": 0 if not (enabled and lead_data.lead_visible) else (1 if gas_override else 2),
    "SET_ME_2": 0x4,
    "SET_ME_3": 0x3,
    "SET_ME_TMP_64": 0x64,
    "DISTANCE_SETTING": hud_control.leadDistanceBars,
  }

  return packer.make_can_msg("SCC_CONTROL", CAN.ECAN, values)


def create_spas_messages(packer, CAN, left_blink, right_blink):
  ret = []

  values = {
  }
  ret.append(packer.make_can_msg("SPAS1", CAN.ECAN, values))

  blink = 0
  if left_blink:
    blink = 3
  elif right_blink:
    blink = 4
  values = {
    "BLINKER_CONTROL": blink,
  }
  ret.append(packer.make_can_msg("SPAS2", CAN.ECAN, values))

  return ret


def create_fca_warning_light(packer, CAN, frame):
  ret = []

  if frame % 2 == 0:
    values = {
      'AEB_SETTING': 0x1,  # show AEB disabled icon
      'SET_ME_2': 0x2,
      'SET_ME_FF': 0xff,
      'SET_ME_FC': 0xfc,
      'SET_ME_9': 0x9,
    }
    ret.append(packer.make_can_msg("ADRV_0x160", CAN.ECAN, values))
  return ret


def create_adrv_messages(packer, CAN, frame):
  # messages needed to car happy after disabling
  # the ADAS Driving ECU to do longitudinal control

  ret = []

  values = {
  }
  ret.append(packer.make_can_msg("ADRV_0x51", CAN.ACAN, values))

  ret.extend(create_fca_warning_light(packer, CAN, frame))

  if frame % 5 == 0:
    values = {
      'SET_ME_1C': 0x1c,
      'SET_ME_FF': 0xff,
      'SET_ME_TMP_F': 0xf,
      'SET_ME_TMP_F_2': 0xf,
    }
    ret.append(packer.make_can_msg("ADRV_0x1ea", CAN.ECAN, values))

    values = {
      'SET_ME_E1': 0xe1,
      'SET_ME_3A': 0x3a,
    }
    ret.append(packer.make_can_msg("ADRV_0x200", CAN.ECAN, values))

  if frame % 20 == 0:
    values = {
      'SET_ME_15': 0x15,
    }
    ret.append(packer.make_can_msg("ADRV_0x345", CAN.ECAN, values))

  if frame % 100 == 0:
    values = {
      'SET_ME_22': 0x22,
      'SET_ME_41': 0x41,
    }
    ret.append(packer.make_can_msg("ADRV_0x1da", CAN.ECAN, values))

  return ret


def hyundai_crc8(data: bytes) -> int:
  poly = 0x2F
  crc = 0xFF
  for byte in data:
    crc ^= byte
    for _ in range(8):
      crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
  return crc ^ 0xFF


def create_steering_messages_adrv(packer, CP, CAN, lat_active, apply_torque, CS, frame):
  ret = []

  # Spoof MDPS to CAM bus — camera needs to see steering state
  if CS.mdps_info:
    values = copy.copy(CS.mdps_info)
    if frame % 1000 < 40:
      values["STEERING_COL_TORQUE"] = values.get("STEERING_COL_TORQUE", 0) + 220
    ret.append(packer.make_can_msg("MDPS", CAN.CAM, values))

  # LFA steering command to ECAN
  values = {
    "LKA_MODE": 2,
    "LKA_ICON": 2 if lat_active else 1,
    "TORQUE_REQUEST": apply_torque,
    "STEER_REQ": 1 if lat_active else 0,
    "HAS_LANE_SAFETY": 0,
    "DAMP_FACTOR": 0 if lat_active else 100,
  }
  ret.append(packer.make_can_msg("LFA", CAN.ECAN, values))

  return ret


def create_acc_control_scc_adrv(packer, CAN, enabled, accel_last, accel, stopping, gas_override, set_speed, hud_control,
                                jerk_u, jerk_l, CS):
  jerk = 5
  jn = jerk / 50
  if not enabled or gas_override:
    a_val, a_raw = 0, 0
  else:
    a_raw = accel
    a_val = np.clip(accel, accel_last - jn, accel_last + jn)

  values = copy.copy(CS.cruise_info)
  values["ACCMode"] = 0 if not enabled else (2 if gas_override else 1)
  values["MainMode_ACC"] = 1
  values["StopReq"] = 1 if stopping else 0
  values["aReqValue"] = a_val
  values["aReqRaw"] = a_raw
  values["VSetDis"] = set_speed
  values["JerkLowerLimit"] = jerk_l if enabled else 1
  values["JerkUpperLimit"] = 2.0 if stopping else jerk_u
  values["DISTANCE_SETTING"] = hud_control.leadDistanceBars

  return packer.make_can_msg("SCC_CONTROL", CAN.ECAN, values)


def create_tcs_messages_adrv(packer, CAN, CS):
  ret = []
  if CS.tcs_info is not None:
    values = copy.copy(CS.tcs_info)
    values["DriverBraking"] = 0
    values["DriverBrakingLowSens"] = 0
    values["NEW_SIGNAL_1"] = 0 if values.get("ACC_REQ") == 1 else 1
    ret.append(packer.make_can_msg("TCS", CAN.ECAN, values))
  return ret


def create_ccnc_messages_adrv(packer, CAN, frame, CC, CS, hud_control):
  from opendbc.car.common.conversions import Conversions as CV
  ret = []

  # ADRV_0x160: Clear LFA fault (every 2 frames)
  if frame % 2 == 0:
    values = {}
    if CS.adrv_160_info is not None:
      values = copy.copy(CS.adrv_160_info)
    values["AEB_SETTING"] = 0x1
    values["SET_ME_2"] = 0x2
    values["SET_ME_FF"] = 0xff
    values["SET_ME_FC"] = 0xfc
    values["SET_ME_9"] = 0x9
    ret.append(packer.make_can_msg("ADRV_0x160", CAN.ECAN, values))

    # Button engagement: auto-engage ACC/LFA when openpilot enables
    if CS.cruise_buttons_msg is not None:
      values = copy.copy(CS.cruise_buttons_msg)
      # Auto-press LDA button to engage LFA if not active
      if CS.LFA_ICON == 0 and 0 < frame % 200 < 12:
        values["LDA_BTN"] = 1
      # Auto-engage ACC if main is on but ACC not active
      if CC.enabled and CS.MainMode_ACC:
        if CS.ACCMode in (0, 4) and 10 < frame % 200 < 22:
          values["CRUISE_BUTTONS"] = 2  # SET_DECEL to engage
      elif CC.enabled and not CS.MainMode_ACC and 10 < frame % 200 <= 16:
        values["ADAPTIVE_CRUISE_MAIN_BTN"] = 1
      ret.append(packer.make_can_msg(CS.cruise_btns_msg_canfd, CAN.ECAN, values))

  # CCNC_0x161: HUD cluster (every 5 frames) — construct from scratch
  if frame % 5 == 0:
    main_enabled = getattr(getattr(CS, 'out', None), 'cruiseState', None)
    main_enabled = main_enabled.available if main_enabled else False
    cruise_enabled = CC.enabled
    lat_active = CC.latActive

    set_speed_in_units = hud_control.setSpeed * (CV.MS_TO_KPH if CS.is_metric else CV.MS_TO_MPH)

    values = {
      "SETSPEED": (3 if cruise_enabled else 1) if main_enabled else 0,
      "SETSPEED_HUD": (3 if cruise_enabled else 1) if main_enabled else 0,
      "SETSPEED_SPEED": int(set_speed_in_units + 0.5),
      "DISTANCE": hud_control.leadDistanceBars,
      "DISTANCE_LEAD": 2 if cruise_enabled and hud_control.leadVisible else 1 if main_enabled and hud_control.leadVisible else 0,
      "DISTANCE_CAR": 2 if cruise_enabled else 1 if main_enabled else 0,
      "HDA_ICON": 2 if cruise_enabled else 1 if main_enabled else 0,
      "LFA_ICON": 2 if lat_active else 1 if main_enabled else 0,
      "LANELINE_LEFT": 2 if hud_control.leftLaneVisible else 0,
      "LANELINE_RIGHT": 2 if hud_control.rightLaneVisible else 0,
    }
    ret.append(packer.make_can_msg("CCNC_0x161", CAN.ECAN, values))

    # CCNC_0x162: Lead vehicle tracking — construct from scratch
    values = {}
    ret.append(packer.make_can_msg("CCNC_0x162", CAN.ECAN, values))

    # ADRV_0x1ea: Side detection passthrough or defaults
    if CS.adrv_1ea_info is not None:
      values = copy.copy(CS.adrv_1ea_info)
    else:
      values = {
        "SET_ME_1C": 0x1c,
        "SET_ME_FF": 0xff,
        "SET_ME_TMP_F": 0xf,
        "SET_ME_TMP_F_2": 0xf,
      }
    ret.append(packer.make_can_msg("ADRV_0x1ea", CAN.ECAN, values))

    # ADRV_0x200: Cruise params
    if CS.adrv_200_info is not None:
      values = copy.copy(CS.adrv_200_info)
    else:
      values = {
        "SET_ME_E1": 0xe1,
        "SET_ME_3A": 0x3a,
      }
    ret.append(packer.make_can_msg("ADRV_0x200", CAN.ECAN, values))

  return ret


def hkg_can_fd_checksum(address: int, sig, d: bytearray) -> int:
  crc = 0
  for i in range(2, len(d)):
    crc = ((crc << 8) ^ CRC16_XMODEM[(crc >> 8) ^ d[i]]) & 0xFFFF
  crc = ((crc << 8) ^ CRC16_XMODEM[(crc >> 8) ^ ((address >> 0) & 0xFF)]) & 0xFFFF
  crc = ((crc << 8) ^ CRC16_XMODEM[(crc >> 8) ^ ((address >> 8) & 0xFF)]) & 0xFFFF
  if len(d) == 8:
    crc ^= 0x5F29
  elif len(d) == 16:
    crc ^= 0x041D
  elif len(d) == 24:
    crc ^= 0x819D
  elif len(d) == 32:
    crc ^= 0x9F5B
  return crc
