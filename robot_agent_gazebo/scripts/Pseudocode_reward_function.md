#Psuedocode to calculate reward: 
"""

import numpy as np

def calc_Reward(ball_pos, plate_pos, plate_angle, control_effort):

    #constants to be tuned
    c1 = 1
    c2 = 1
    c3 = 1
    

    #calculated reward of ball position
    x = ball_pos_x - plate_pos_x
    y = ball_pos_y - plate_pos_y
    z = ball_pos_z - plate_pos_z
    r_ball = np.exp(-c1*(x^2 + y^2 + z^2))           #larger penealty for larger distances from center of plate
    

    #calculated reward of plate angle
    Phi = plate_pos_roll
    Theta = plate_pos_pitch
    Psi = plate_pos_yaw
    r_plate = -c2*(Phi^2 + Theta^2 + Psi^2)          #penalty for large plate angles


    #calculated reward of control effort
    r_action = -c3*(square and sum all torques)      #penalty for torque commands


    #Total Reward
    R = r_ball + r_plate + r_action


"""
