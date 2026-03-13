from joystick import Joystick
import map_plane_widget as mpw

CMU_LAT =  40.4446
CMU_LON = -79.9440
ZOOM    =  11

if __name__ == '__main__':
    joystick = Joystick()
    try:
        mpw.run(lat=CMU_LAT, lon=CMU_LON, zoom=ZOOM, joystick=joystick)
    finally:
        joystick.close()