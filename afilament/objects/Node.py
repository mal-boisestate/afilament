import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from termcolor import colored, cprint


from afilament.objects.SingleFiber import SingleFiber


class Node:

    def __init__(self, x, y, z, cnt, actin_id):
        self.x = x
        self.y = y
        self.z = z
        self.n = 1
        self.cnt = cnt
        self.actin_ids = [actin_id]
        self.geodesic_distance_from_center = None
        self.tg_xy = None
        self.ideal_x = None
        self.ideal_y = None
        self.ideal_z = None
        self.quadrant = None

    def get_stat(self, scale_y, scale_z):
        return [self.x, self.y, self.z,
                ",".join(map(str, self.actin_ids) ),
                cv2.contourArea(self.cnt) * scale_y * scale_z]

    def add_distance_from_center(self, geodesic_distance):
        self.geodesic_distance_from_center = geodesic_distance

    def add_xy_direction(self, tg_betta, quadrant):
        self.tg_xy = tg_betta
        self.quadrant = quadrant

    def add_ideal_coordinates(self, x, y, z):
        self.ideal_x = x
        self.ideal_y = y
        self.ideal_z = z

def plot_connected_nodes(actin_fibers, nodes, pairs, user_title="", plot_nodes=True):
    ax = plt.axes(projection='3d')
    # ax = plt.axes()  #for 2D testing
    for fiber in actin_fibers:
        if len(np.unique(fiber.zs)) < 1:
            continue

        # Draw only center points
        xdata = fiber.xs
        ydata = fiber.ys
        zdata = fiber.zs
        color_x = 1.0 * np.random.randint(255) / 255
        color_y = 1.0 * np.random.randint(255) / 255
        color_z = 1.0 * np.random.randint(255) / 255
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), cmap='Greens')

    if plot_nodes:
        xdata, ydata, zdata = [], [], []
        for node in nodes:
            xdata.append(node.x)
            ydata.append(node.y)
            zdata.append(node.z)
        color_x = 1.0
        color_y = 0
        color_z = 0
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), s=50, cmap='Greens')

    for right_id, left_id in pairs:
        right_node, left_node = nodes[right_id], nodes[left_id]
        plt.plot((right_node.x, left_node.x), (right_node.y, left_node.y), (right_node.z, left_node.z), color='black')
    plt.title(f"Show fibers connections \n {user_title}")
    plt.show()

def plot_nodes(actin_fibers, nodes):
    ax = plt.axes(projection='3d')
    # ax = plt.axes()  #for 2D testing
    for fiber in actin_fibers:
        if len(np.unique(fiber.zs)) < 1:
            continue

        # Draw only center points
        xdata = fiber.xs
        ydata = fiber.ys
        zdata = fiber.zs
        color_x = 1.0 * np.random.randint(255) / 255
        color_y = 1.0 * np.random.randint(255) / 255
        color_z = 1.0 * np.random.randint(255) / 255
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), cmap='Greens')

    xdata, ydata, zdata = [], [], []
    for node in nodes:
        xdata.append(node.x)
        ydata.append(node.y)
        zdata.append(node.z)
    color_x = 1.0
    color_y = 0
    color_z = 0
    if xdata:
        ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), s=50, cmap='Greens')

    plt.show()

def plot_branching_nodes(actin_fibers, nodes, min_fiber_length, resolution,
                         structure, img_num, cell_num):
    """
    Plot actin fibers and branching nodes in 3D space.

    Parameters:
        actin_fibers (object): ActinFiber objects.
        nodes (list): A list of Node objects.
    """

    def plot_actin_fibers(fig_ax):
        """
        Plot actin fibers in 3D space.
        """

        # Set up colors for each fiber
        colors = np.random.rand(len(actin_fibers_filtered), 3)
        reference_xsection_um = 30

        # Plot each fiber
        temp = 0
        for i, fiber in enumerate(actin_fibers_filtered):
            if len(np.unique(fiber.zs)) < 1:
                continue

            # Draw only center points
            xdata = fiber.xs
            ydata = fiber.ys
            zdata = fiber.zs
            fiber_weight = np.mean([cv2.contourArea(cnt) for cnt in fiber.cnts]) / reference_xsection_um

            if xdata:
                fig_ax.scatter3D(
                    xdata, ydata, zdata,
                    c=[colors[i]], cmap='Greens', s=100*fiber_weight, alpha=0.2
                )

                # This code maps the correspondence between fiber color and fiber number
                # in the statistical file. It is currently commented out since the data is not necessary.

                # Uncomment the following code to display fiber information:
                # for i, fiber in enumerate(fibers):
                #     start = (fiber.xs[0], fiber.ys[0])
                #     end = (fiber.xs[-1], fiber.ys[1])
                #     print(f"I am fiber {i}, I start at {start}, end at {end}")
                #
                #     # Add text label to plot
                #     fig_ax.text(
                #         -0.2, temp,
                #         f"I am fiber {i}",
                #         size=12,
                #         color=colors[i]
                #     )
                #     temp += 0.005


    def plot_branching_nodes(fig_ax):
        """
        Plot branching nodes in 3D space.
        """
        # Collect data for branching nodes
        xdata, ydata, zdata = [], [], []

        # Branching is a node that has more than one actin
        for node in branching_nodes:
            if len(node.actin_ids) > 1:
                xdata.append(node.x)
                ydata.append(node.y)
                zdata.append(node.z)

        # Set up color for branching nodes
        color = [1.0, 0, 0]

        # Plot branching nodes
        if xdata:
            fig_ax.scatter3D(xdata, ydata, zdata, c=[color] * len(xdata), s=100, cmap='Greens', depthshade=False)


    actin_fibers_filtered = [fiber for fiber in actin_fibers.fibers_list if fiber.n >= min_fiber_length]
    branching_nodes = [node for node in nodes if len(node.actin_ids) > 1]

    # Set up figure
    fig = plt.figure()
    fig_ax = fig.add_subplot(111, projection='3d')

    # Plot actin fibers and branching nodes
    plot_actin_fibers(fig_ax)
    plot_branching_nodes(fig_ax)

    # Add annotations
    fig_ax.text2D(
        1.05, 0.60,
        f"Min fiber len threshold: {min_fiber_length * resolution.x:.2f} \u03BCm\n"
        f"Fiber number: {actin_fibers.total_num}\n"
        f"Fiber length (total): {actin_fibers.total_length:.2f} \u03BCm\n"
        f"Fiber volume (total): {actin_fibers.total_volume:.2f} $\u03BCm^3$\n"
        f"Fiber intensity (total): {actin_fibers.intensity/10**6:.0f} * $10^6$\n"
        f"Fiber intensity (all f-actin signal): {actin_fibers.f_actin_signal_total_intensity/10**6:.0f} * $10^6$\n"
        f"Nurbanu branching coef.: N/A\n"
        f"Total nodes: {len(nodes)}\n"
        f"Branching nodes: {len(branching_nodes)}\n",
        transform=fig_ax.transAxes,
        linespacing=2,
        size=11,
        bbox=dict(boxstyle="square,pad=0.5", fc="lightblue")
    )

    plt.title(f"Actin fiber {structure} of \n"
              f"image # {img_num}, cell # {cell_num}")

    # Show plot
    plt.show()


def add_edge_nodes(actin_fibers):
    nodes = []
    for actin_id, actin in enumerate(actin_fibers):
        nodes.append(Node(actin.xs[0], actin.ys[0], actin.zs[0], actin.cnts[0], actin_id))  # left side
        nodes.append(Node(actin.xs[-1], actin.ys[-1], actin.zs[-1], actin.cnts[-1], actin_id))  # right side
        actin.left_node_id = len(nodes) - 2
        actin.right_node_id = len(nodes) - 1
    return actin_fibers, nodes


def find_branching_nodes(fibers, new_actin_len_th, is_plot_nodes=False):
    """
    new_actin_len_th - do not breake actin if one of the part is too small
    """
    actin_fibers = copy.deepcopy(fibers)
    actin_fibers, nodes = add_edge_nodes(actin_fibers)

    for left_node in nodes[::2]:
        actin_id_to_break = None

        actins_to_check = [[actin_id, actin] for actin_id, actin in enumerate(actin_fibers)
                           if left_node.x - 1 in actin.xs]
        cnts_to_check = [actin.cnts[actin.xs.index(left_node.x - 1)] for _, actin in actins_to_check]

        node_mask = np.zeros((1300, 1300), dtype=np.uint8)
        cv2.drawContours(node_mask, [left_node.cnt], -1, 255, -1)
        for cnt_id, cnt in enumerate(cnts_to_check):
            cnt_mask = np.zeros((1300, 1300), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            if np.any(cv2.bitwise_and(node_mask, cnt_mask)):
                actin_id_to_break = actins_to_check[cnt_id][0]

        if actin_id_to_break is not None:
            actin_to_break = actin_fibers[actin_id_to_break]
            break_index = actin_to_break.xs.index(left_node.x - 1)

            # do not break actin if one of the parts is too small
            if break_index < new_actin_len_th or len(actin_to_break.xs) - new_actin_len_th < break_index:
                continue

            new_right_actin = SingleFiber(-1, -1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)
            new_right_actin.left_node_id = nodes.index(left_node)
            new_right_actin.right_node_id = actin_to_break.right_node_id

            actin_to_break.xs = actin_to_break.xs[:break_index + 1]
            actin_to_break.ys = actin_to_break.ys[:break_index + 1]
            actin_to_break.zs = actin_to_break.zs[:break_index + 1]
            actin_to_break.last_layer = actin_to_break.last_layer[:break_index + 1]
            actin_to_break.cnts = actin_to_break.cnts[:break_index + 1]
            actin_to_break.n = len(actin_to_break.xs)

            actin_fibers.append(new_right_actin)

            # update old actin
            actin_to_break.right_node_id = nodes.index(left_node)

            # update node
            left_node.actin_ids.append(actin_id_to_break)  # adding old actin which attached from left
            left_node.actin_ids.append(len(actin_fibers) - 1)  # adding new actin which attached from right

            # update old right node
            old_right_node = nodes[new_right_actin.right_node_id]
            old_right_node.actin_ids.pop(old_right_node.actin_ids.index(actin_id_to_break))
            old_right_node.actin_ids.append(len(actin_fibers) - 1)

    # -------------------------------------------------------- #
    for right_node in nodes[1::2]:
        actin_id_to_break = None

        actins_to_check = [[actin_id, actin] for actin_id, actin in enumerate(actin_fibers)
                           if right_node.x + 1 in actin.xs]
        cnts_to_check = [actin.cnts[actin.xs.index(right_node.x + 1)] for _, actin in actins_to_check]

        node_mask = np.zeros((1300, 1300), dtype=np.uint8)
        cv2.drawContours(node_mask, [right_node.cnt], -1, 255, -1)
        for cnt_id, cnt in enumerate(cnts_to_check):
            cnt_mask = np.zeros((1300, 1300), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            if np.any(cv2.bitwise_and(node_mask, cnt_mask)):
                actin_id_to_break = actins_to_check[cnt_id][0]

        if actin_id_to_break is not None:
            actin_to_break = actin_fibers[actin_id_to_break]
            break_index = actin_to_break.xs.index(right_node.x + 1)

            # do not break actin if one of the parts is too small
            if break_index < new_actin_len_th or len(actin_to_break.xs) - new_actin_len_th < break_index:
                continue

            new_right_actin = SingleFiber(-1, -1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)
            new_right_actin.left_node_id = nodes.index(right_node)
            new_right_actin.right_node_id = actin_to_break.right_node_id

            actin_to_break.xs = actin_to_break.xs[:break_index + 1]
            actin_to_break.ys = actin_to_break.ys[:break_index + 1]
            actin_to_break.zs = actin_to_break.zs[:break_index + 1]
            actin_to_break.last_layer = actin_to_break.last_layer[:break_index + 1]
            actin_to_break.cnts = actin_to_break.cnts[:break_index + 1]
            actin_to_break.n = len(actin_to_break.xs)

            actin_fibers.append(new_right_actin)

            # update old actin
            actin_to_break.right_node_id = nodes.index(right_node)

            # update node
            right_node.actin_ids.append(actin_id_to_break)  # adding old actin which attached from left
            right_node.actin_ids.append(len(actin_fibers) - 1)  # adding new actin which attached from right

            # update old right node
            old_right_node = nodes[new_right_actin.right_node_id]
            old_right_node.actin_ids.pop(old_right_node.actin_ids.index(actin_id_to_break))
            old_right_node.actin_ids.append(len(actin_fibers) - 1)
    if is_plot_nodes:
        plot_nodes(actin_fibers, nodes)
    return nodes, actin_fibers
