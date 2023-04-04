import numpy as np
import math


class MergedFiber(object):
    # scale_x = 0.04  # units - micrometer
    # scale_y = 0.04  # units - micrometer (image was resized to account for different scale along z axis)
    # scale_z = 0.17  # units - micrometer (image was resized to account for different scale along z axis)

    def __init__(self, fiber, fiber_id):
        self.fibers = [fiber]
        self.fibers_ids = [fiber_id]
        self.gaps_distance = 0

    def merge(self, another_fiber, left_node, right_node, resolution):
        self.fibers.extend(another_fiber.fibers)
        self.fibers_ids.extend(another_fiber.fibers_ids)
        # distance = np.sqrt((left_node.x - right_node.x) ** 2 + (left_node.y - right_node.y) ** 2 + (left_node.z - right_node.z) ** 2))
        distance_between_given_fibers = np.sqrt((left_node.x - right_node.x)**2 +
                           (left_node.y - right_node.y)**2 +
                           ((left_node.z - right_node.z) * (resolution.z/resolution.y))**2)
        self.gaps_distance = self.gaps_distance + another_fiber.gaps_distance + distance_between_given_fibers


    def get_stat(self, resolution):
        """
        merged_actin_length_with_gaps - full length of fiber pieces plus gaps_distance
        merged_actin_xsection - avarage actin xsection
        merged_actin_volume_with_gaps - actin gap distance times average xsection plus total volume of actual fibers
        merged_n - total number of layers with actual fibers
        slope
        """
        merged_actin_length = 0
        merged_n = 0
        merged_actin_volume = 0
        for fiber in self.fibers:
            actin_length, _, actin_volume, n, _ = fiber.get_stat(resolution)
            merged_actin_length += actin_length
            merged_n += n
            merged_actin_volume += actin_volume

        merged_actin_xsection = merged_actin_volume / merged_actin_length
        merged_actin_length_with_gaps = merged_actin_length + self.gaps_distance * resolution.x
        merged_actin_volume_with_gaps = merged_actin_volume + self.gaps_distance * resolution.x * merged_actin_xsection

        adjacent = (self.fibers[-1].xs[-1] - self.fibers[0].xs[0])
        if adjacent == 0:
            adjacent = 1

        slope = math.degrees(math.tan((self.fibers[-1].ys[-1] - self.fibers[0].ys[0]) / adjacent))

        return [merged_actin_length_with_gaps, merged_actin_xsection, merged_actin_volume_with_gaps, merged_n, slope]

