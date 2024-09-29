import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

from objects.SingleFiber import SingleFiber

reference_xsection_um = 40

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
                ",".join(map(str, self.actin_ids)),
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


def plot_branching_nodes(actin_fibers, nodes, min_fiber_thr_microns, resolution,
                         structure, img_num, cell_num, nuc_center, img_name):
    """
    Plot actin fibers and branching nodes in 3D space.

    Parameters:
        actin_fibers (object): ActinFiber objects.
        nodes (list): A list of Node objects.
    """

    def find_closest_index(numbers, target):
        if not numbers:  # Return None if the list is empty
            return None

        closest_index = 0  # Start with the index 0
        closest_difference = abs(target - numbers[0])  # Compute the difference

        # Loop through the rest of the list
        for index, number in enumerate(numbers[1:], 1):
            difference = abs(target - number)
            # If the current number is closer, update closest_index
            if difference < closest_difference:
                closest_index = index
                closest_difference = difference

        return closest_index

    def plot_actin_fibers(fig_ax):
        """
        Plot actin fibers in 3D space.
        """

        # Set up colors for each fiber
        colors = np.random.rand(len(actin_fibers_filtered), 3)

        fixed_size = 10

        # Plot each fiber
        temp = -0.1
        text_elements = []

        # Calculate the maximum y-coordinate for flipping along y-axis
        all_ys = [y for fiber in actin_fibers_filtered for y in fiber.ys if len(fiber.ys) > 0]
        max_y = max(all_ys)

        # Calculate the maximum x-coordinate for flipping along x-axis
        all_xs = [x for fiber in actin_fibers_filtered for x in fiber.xs if len(fiber.xs) > 0]
        max_x = max(all_xs)

        for i, fiber in enumerate(actin_fibers_filtered):
            if len(np.unique(fiber.zs)) < 1:
                continue

            # Draw only center points
            xdata = [max_x - x for x in fiber.xs] # Flip x-coordinates
            ydata = fiber.ys # Flip y-coordinates
            zdata = fiber.zs
            fiber_weight = np.mean([cv2.contourArea(cnt) for cnt in fiber.cnts]) / reference_xsection_um

            if xdata:
                fig_ax.scatter3D(
                    xdata, ydata, zdata,
                    c=[colors[i]], cmap='Greens', s=20*fiber_weight, alpha=0.2,
                )  # alternatively s=20*fiber_weight or s=fixed_size

            # This code maps the correspondence between fiber color and fiber number
            # in the statistical file. It is currently commented out since the data is not necessary.

            # Add text label to plot
            # This part of code helps match fiber color and fiber statistics. Is optional.Comment out
            # old_len = (fiber.xs[-1] - fiber.xs[0]) * resolution.x
            # old_xsection = np.mean([cv2.contourArea(cnt) for cnt in fiber.cnts]) * resolution.y * resolution.z
            # old_volume = old_len * old_xsection

            # len_each_1 = fiber.get_length_test_k(resolution, 1)
            # len_each_2 = fiber.get_length_test_k(resolution, 2)
            # len_each_3 = fiber.get_length_test_k(resolution, 3)
            # len_each_5 = fiber.get_length_test_k(resolution, 5)
            actin_center_indx = find_closest_index(fiber.xs, nuc_center[0])

            # Create and store the text element
            text_element = fig_ax.text2D(
                -0.2, temp,
                f"Length: {fiber.length:.2f} μm | Volume: {fiber.volume:.2f} μm^3\n"
                # f"Center* x:{fiber.xs[actin_center_indx]} y:{fiber.ys[actin_center_indx]}  z:{fiber.zs[actin_center_indx]} \n"
                # f"Start x:{max_x - fiber.xs[-1]} y:{fiber.ys[-1]} z:{fiber.zs[-1]} | End x:{max_x - fiber.xs[0]} y:{fiber.ys[0]} z:{fiber.zs[0]}\n"
            ,
                size=12,
                color=colors[i]
            )
            text_elements.append(text_element)
            temp += 0.010

        def scroll_event_handler(event):
            # Adjust y-coordinates of text elements based on scroll direction
            for text in text_elements:
                x, y = text.get_position()
                y += event.step * 0.005  # Adjust this value to control scroll speed
                text.set_position((x, y))

            fig_ax.figure.canvas.draw_idle()

        # Connect the scroll event to the handler
        fig_ax.figure.canvas.mpl_connect('scroll_event', scroll_event_handler)

    def plot_green_actin_fibers(fig_ax):
        """
        Plot actin fibers in 3D space with a uniform bright green color and size.
        """
        bright_green_color = [0, 1, 0]  # RGB for bright green
        bright_red_color = [1, 0.3, 0]  # RGB for bright green
        fixed_size = 10  # Fixed size for all fibers
        all_xs = [x for fiber in actin_fibers_filtered for x in fiber.xs if len(fiber.xs) > 0]
        max_x = max(all_xs)


        # Plot each fiber
        for fiber in actin_fibers_filtered:
            if len(np.unique(fiber.zs)) < 1:
                continue

            # Draw only center points
            xdata = [max_x - x for x in fiber.xs]  # Flip x-coordinates
            ydata = fiber.ys  # Flip y-coordinates
            zdata = fiber.zs
            fiber_weight = np.mean([cv2.contourArea(cnt) for cnt in fiber.cnts]) / reference_xsection_um


            if xdata:
                fig_ax.scatter3D(
                    xdata, ydata, zdata,
                    color=bright_green_color, s=20*fiber_weight, alpha=0.6
                ) #s=fixed_size or s=20*fiber_weight

    def plot_branching_nodes(fig_ax):
        """
        Plot branching nodes in 3D space.
        """
        # Collect data for branching nodes
        xdata, ydata, zdata = [], [], []

        # Calculate the maximum x-coordinate for flipping along x-axis
        all_xs = [x for fiber in actin_fibers_filtered for x in fiber.xs if len(fiber.xs) > 0]
        max_x = max(all_xs)

        # Branching is a node that has more than one actin
        for node in branching_nodes:
            if len(node.actin_ids) > 1:
                xdata.append(max_x - node.x)
                ydata.append(node.y)
                zdata.append(node.z)

        # Set up color for branching nodes
        color = [1.0, 0, 0]

        # Plot branching nodes
        if xdata:
            fig_ax.scatter3D(xdata, ydata, zdata, c=[color] * len(xdata), s=25, cmap='Greens', depthshade=False)

    actin_fibers_filtered = [fiber for fiber in actin_fibers.fibers_list if fiber.length >= min_fiber_thr_microns]
    branching_nodes = [node for node in nodes if len(node.actin_ids) > 1]

    # ###################################CHANGES FOR MADISON FIGURES####################################
    # #################THIS PART OF CODE CAN BE REMOVED AFTER#############################################
    # # Set up figure with a transparent background
    # fig = plt.figure()
    # fig.patch.set_alpha(0.0)  # Set the transparency for the figure's background
    # fig_ax = fig.add_subplot(111, projection='3d')
    # fig_ax.patch.set_alpha(0.0)  # Set the transparency for the axes' background
    #
    # # Remove the gridlines and ticks
    # fig_ax.grid(False)  # Turn off the gridlines
    # fig_ax.xaxis._axinfo["grid"]['color'] = (0, 0, 0, 0)
    # fig_ax.yaxis._axinfo["grid"]['color'] = (0, 0, 0, 0)
    # fig_ax.zaxis._axinfo["grid"]['color'] = (0, 0, 0, 0)
    #
    # # Turn off the pane color to make it transparent as well
    # fig_ax.xaxis.pane.fill = False
    # fig_ax.yaxis.pane.fill = False
    # fig_ax.zaxis.pane.fill = False
    #
    # # Optionally, you can also make the axis pane color transparent
    # fig_ax.xaxis.pane.set_edgecolor('w')
    # fig_ax.yaxis.pane.set_edgecolor('w')
    # fig_ax.zaxis.pane.set_edgecolor('w')
    #
    # # Remove axis lines and ticks
    # fig_ax.xaxis.line.set_visible(False)
    # fig_ax.yaxis.line.set_visible(False)
    # fig_ax.zaxis.line.set_visible(False)
    # fig_ax.set_xticks([])
    # fig_ax.set_yticks([])
    # fig_ax.set_zticks([])
    #
    # ###################################################################

    # Set up figure
    fig = plt.figure()
    fig_ax = fig.add_subplot(111, projection='3d')

    # Plot actin fibers and branching nodes
    plot_green_actin_fibers(fig_ax) #Modified function for Madisons's figures
    # plot_actin_fibers(fig_ax)

    # plot_branching_nodes(fig_ax)

    # Add annotations
    fig_ax.text2D(
        1.05, 0.5,
        f"Min fiber length threshold: {min_fiber_thr_microns:.2f} \u03BCm\n"
        f"Fiber number: {actin_fibers.total_num}\n"
        f"Fiber length (total): {actin_fibers.total_length:.2f} \u03BCm\n"
        f"Fiber volume (total): {actin_fibers.total_volume:.2f} $\u03BCm^3$\n"
        f"Fiber intensity (total): {actin_fibers.intensity / 10 ** 6:.0f} * $10^6$\n"
        f"Fiber intensity (all f-actin signal): {actin_fibers.f_actin_signal_total_intensity / 10 ** 6:.0f} * $10^6$\n"
        f"Total nodes number: {len(nodes)}\n"
        f"Branching nodes number: {len(branching_nodes)}\n"
        # f"Nucleus center (pixels) x: {nuc_center[0]} y:{nuc_center[1]}\n"
        # f"*The term 'center actin fiber coordinates' does\n"
        # f"not refer to the geometric center of the actin.\n"
        # f"Instead, it denotes the point on the actin fiber\n"
        # f"that is nearest to the nucleus's center, based \n"
        # f"solely on the x-coordinate.\n"
        ,
        transform=fig_ax.transAxes,
        linespacing=2,
        size=11,
        bbox=dict(boxstyle="square,pad=0.5", fc="lightblue")
    )

    structure_name = None
    if structure == "cap":
        structure_name = "Apical"
    elif structure == "bottom":
        structure_name = "Basal"
    else:
        structure_name = ""

    plt.title(f"{structure_name} Actin Fiber of image # {img_num}, cell # {cell_num} \n"
              f"{img_name}")

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
