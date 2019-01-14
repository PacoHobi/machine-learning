import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter_3d(results, episodes):
    """3D scatterplot of the position of the drone over multiple episodes."""
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for episode in episodes:
        x_vals = results['x'][episode]
        y_vals = results['y'][episode]
        z_vals = results['z'][episode]
        c_vals = range(len(x_vals))
        ax.scatter(x_vals, y_vals, z_vals, c=c_vals, cmap='winter')


def plot_episodes(results, scores, task, episodes, cols=4):
    """
    Plot the position, distance to the init position and rotor speeds of the 
    drone for multiple episodes.
    """
    if not isinstance(episodes, (list, tuple)):
        episodes = list(episodes)

    num_episodes = len(episodes)

    plt.rcParams['figure.figsize'] = [15, min(num_episodes / cols * 4, 910)]
    gs = gridspec.GridSpec(num_episodes // cols + 1, cols)
    gs.update(wspace=0.4, hspace=0.3)

    for idx, episode in enumerate(episodes):
        ax = plt.subplot(gs[idx // cols, idx % cols])
        ax.set_title('Episode %d (score %d)' % (episode, scores[episode]))
        ax.set_xlim([0, task.sim.runtime])
        ax.set_ylim([-2 * task.target_pos[2], 2 * task.target_pos[2]])
        ax.plot(results['time'][episode], results['x'][episode], label='x')
        ax.plot(results['time'][episode], results['y'][episode], label='y')
        ax.plot(results['time'][episode], results['z'][episode], 'r', label='z')
        #ax.plot(results['time'][episode], results['distance'][episode], 'm', label='distance')
        ax.legend()
        
        ax2 = ax.twinx()
        ax2.set_ylim([task.action_low - 100, task.action_high + 100])
        ax2.plot(results['time'][episode], results['rotor_speed1'][episode], '--', linewidth=1.)
        ax2.plot(results['time'][episode], results['rotor_speed2'][episode], '--', linewidth=1.)
        ax2.plot(results['time'][episode], results['rotor_speed3'][episode], '--', linewidth=1.)
        ax2.plot(results['time'][episode], results['rotor_speed4'][episode], '--', linewidth=1.)