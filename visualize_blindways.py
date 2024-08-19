'''
Visualization
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import textwrap

mpl.rcParams['font.family'] = 'Calibri'
# COLOR_DICT = {'Left': "#4ec36b", 'Center': 'k', 'Right': "#2c718e", 'Cane' : 'k'}
COLOR_DICT = {'Left': '#DB9CC6', 'Center': 'k', 'Right': '#B6508C', 'Cane' : 'w'}

HAND_DICT = {'al': 'left', 'bill': 'right', 'brian': 'right', 'bruce': 'right', 'dany': 'right', 'dino': 'left', 'kayla': 'right', 'marhsa': 'right', 'matthew': 'right', 'minh': 'right', 'nick': 'right'}
CANE_AVG_LENGTH = 0.7

""" 
Xsens
"""

XSENS_JOINT_ORDER= ['Pelvis',
'L5','L3','T12', 'T8',
'Neck','Head',
'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',
'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',
'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe',
'Prop']

XSENS_JOINT_PAIRS = [
    ("center", "Pelvis", "L5"),
    ("center", "L5", "L3"),
    ("center", "L3", "T12"),
    ("center", "T12", "T8"),
    ("center", "Head", "Neck"),
    ("center", "Neck", "T8"),
    ("right", "T8", "Right Shoulder"), 
    ("left","T8", "Left Shoulder"),
    ("right","Right Shoulder", "Right Upper Arm"),
    ("right","Right Upper Arm", "Right Forearm"),
    ("right","Right Forearm", "Right Hand"),
    ("left","Left Shoulder", "Left Upper Arm"),
    ("left","Left Upper Arm", "Left Forearm"),
    ("left","Left Forearm", "Left Hand"),
    ("right","Pelvis", "Right Upper Leg"),
    ("right","Right Upper Leg", "Right Lower Leg"),
    ("right","Right Lower Leg", "Right Foot"),
    ("right","Right Foot", "Right Toe"),
    ("left","Pelvis", "Left Upper Leg"),
    ("left","Left Upper Leg", "Left Lower Leg"),
    ("left","Left Lower Leg", "Left Foot"),
    ("left","Left Foot", "Left Toe"),
]

def PLOT_CARLA_3D_FINAL(SAVE_DIR, kpts3d, XSENS_JOINT_ORDER_w_cane_extended, XSENS_JOINT_PAIRS_w_cane_extended):
    LColor = COLOR_DICT['Left']
    RColor = COLOR_DICT['Right']
    CColor = COLOR_DICT['Cane']

    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black') 
    ax = plt.subplot(projection='3d', facecolor='black')

    # kpts3d[:, 1], kpts3d[:, 2] = kpts3d[:, 2], kpts3d[ :, 1].copy()

    for idx, (side, sname, ename) in enumerate(XSENS_JOINT_PAIRS_w_cane_extended):
        sidx = XSENS_JOINT_ORDER_w_cane_extended.index(sname)
        eidx = XSENS_JOINT_ORDER_w_cane_extended.index(ename)

        sjoint = kpts3d[sidx]
        ejoint = kpts3d[eidx]

        rpts = np.stack((sjoint, ejoint), axis=1)

        if side == 'right':
            tcolor = RColor
        elif side == 'cane':
            tcolor = CColor
        else:
            tcolor = RColor
            
        ax.plot(rpts[0, :], rpts[1, :], rpts[2, :], '-', color=tcolor, alpha=0.95, lw=6, solid_capstyle="round", solid_joinstyle="round")
        # ax.plot(rpts[0, :], -rpts[1, :], rpts[2, :], 'x-', color=tcolor, alpha=0.95, lw=6, solid_capstyle="round", solid_joinstyle="round")

        
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Get rid of the lines and plane
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.xaxis.line.set_color(white)
    ax.yaxis.line.set_color(white)
    ax.zaxis.line.set_color(white)
    
    # Remove the grid and set the background color
    ax.grid(False)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
    plt.tight_layout(pad=0)
    plt.savefig(SAVE_DIR, bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close()

def get_joint_pair_between_cane_and_ground(motion, joint_order, joint_pairs, hand_side):

    motion = motion.copy()  # Ensure we don't modify the original motion array
    joint_order = joint_order.copy()
    joint_pairs = joint_pairs.copy()  # Ensure we don't modify the original joint_pairs

    if hand_side == 'right':
        
        last_joint = motion[:, joint_order.index("Prop")]
        last_2_joint = motion[:, joint_order.index("Right Hand")]
        direction = last_joint - last_2_joint
        direction = direction / np.linalg.norm(direction, axis=1)[:, None] * CANE_AVG_LENGTH
        new_joint = last_joint + direction
        motion = np.concatenate((motion, new_joint[:, None, :]), axis=1)
        joint_pairs.append(('cane', 'Right Hand', 'Prop'))
        joint_pairs.append(('cane', 'Prop', 'Prop_extended'))
        joint_order.append('Prop_extended')
        
    else:  # if hand_side == 'left'
        
        last_joint = motion[:, joint_order.index("Prop")]
        last_2_joint = motion[:, joint_order.index("Left Hand")]
        direction = last_joint - last_2_joint
        direction = direction / np.linalg.norm(direction, axis=1)[:, None] * CANE_AVG_LENGTH
        new_joint = last_joint + direction
        motion = np.concatenate((motion, new_joint[:, None, :]), axis=1)
        joint_pairs.append(('cane', 'Left Hand', 'Prop'))
        joint_pairs.append(('cane', 'Prop', 'Prop_extended'))
        joint_order.append('Prop_extended')
        
    return motion, joint_order, joint_pairs

import numpy as np
import os
import subprocess

def save_video(input_file_name, output_file_name):
    # Ensure the input is a glob pattern or a single file
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-pattern_type', 'glob',
        '-i', input_file_name,  # Directly pass the filename variable
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=20",  # Video filter for scale and fps
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        output_file_name
    ]
    
    try:
        # Execute the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Video is saved as '{output_file_name}'")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed to process the file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


LColor = COLOR_DICT['Left']
RColor = COLOR_DICT['Right']
CColor = COLOR_DICT['Cane']

datadir = 'BlindWays/Motion' # Data directory to visualise
all_filename = os.listdir(datadir)

for seq_filename in all_filename:
    

    seq_name = seq_filename[:-4]
    seq_dir = f'{datadir}/{seq_name}.npy'
    seq = np.load(seq_dir)
    seq = seq[::3] #downsample!!!!!!!!, so fps = 20 up there
    
    os.makedirs(f'visualisation/{seq_name}', exist_ok = True) #Save directory for visualisation

    motion_w_cane_extended = []
    XSENS_JOINT_ORDER_w_cane_extended = []
    XSENS_JOINT_PAIRS_w_cane_extended = []
        
    if ('al' in seq_name) or ('dino' in seq_name):
        motion_w_cane_extended, XSENS_JOINT_ORDER_w_cane_extended_lhanded, XSENS_JOINT_PAIRS_w_cane_extended_lhanded = get_joint_pair_between_cane_and_ground(seq, XSENS_JOINT_ORDER, XSENS_JOINT_PAIRS, 'left')
        XSENS_JOINT_ORDER_w_cane_extended = XSENS_JOINT_ORDER_w_cane_extended_lhanded
        XSENS_JOINT_PAIRS_w_cane_extended = XSENS_JOINT_PAIRS_w_cane_extended_lhanded

    else:
        motion_w_cane_extended, XSENS_JOINT_ORDER_w_cane_extended_rhanded, XSENS_JOINT_PAIRS_w_cane_extended_rhanded = get_joint_pair_between_cane_and_ground(seq, XSENS_JOINT_ORDER, XSENS_JOINT_PAIRS, 'right')
        XSENS_JOINT_ORDER_w_cane_extended = XSENS_JOINT_ORDER_w_cane_extended_rhanded
        XSENS_JOINT_PAIRS_w_cane_extended = XSENS_JOINT_PAIRS_w_cane_extended_rhanded
    
    for t in range(motion_w_cane_extended.shape[0]):
        
        kpts3d = motion_w_cane_extended[t]
        
        """ 
        PLOT 
        """
        fig = plt.figure(facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black') 
        ax = plt.subplot(projection='3d', facecolor='black')

        # kpts3d[:, 1], kpts3d[:, 2] = kpts3d[:, 2], kpts3d[ :, 1].copy()

        for idx, (side, sname, ename) in enumerate(XSENS_JOINT_PAIRS_w_cane_extended):
            sidx = XSENS_JOINT_ORDER_w_cane_extended.index(sname)
            eidx = XSENS_JOINT_ORDER_w_cane_extended.index(ename)

            sjoint = kpts3d[sidx]
            ejoint = kpts3d[eidx]

            rpts = np.stack((sjoint, ejoint), axis=1)

            if side == 'right':
                tcolor = RColor
            elif side == 'cane':
                tcolor = CColor
            else:
                tcolor = RColor
                
            ax.plot(rpts[0, :], rpts[1, :], rpts[2, :], '-', color=tcolor, alpha=0.95, lw=6, solid_capstyle="round", solid_joinstyle="round")
            # ax.plot(rpts[0, :], -rpts[1, :], rpts[2, :], 'x-', color=tcolor, alpha=0.95, lw=6, solid_capstyle="round", solid_joinstyle="round")

            
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Get rid of the lines and plane
        white = (1.0, 1.0, 1.0, 0.0)
        ax.xaxis.set_pane_color(white)
        ax.yaxis.set_pane_color(white)
        ax.zaxis.set_pane_color(white)

        ax.xaxis.line.set_color(white)
        ax.yaxis.line.set_color(white)
        ax.zaxis.line.set_color(white)
        
        # Remove the grid and set the background color
        ax.grid(False)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
        plt.tight_layout(pad=0)
        plt.savefig(f'visualization/{seq_name}/{t:012d}.jpg', bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()

        
    save_video(f'visualization/{seq_name}/*.jpg', f'visualization/{seq_name}.mp4')

