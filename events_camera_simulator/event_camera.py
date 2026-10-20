import numpy as np
import os

class EventCameraSimulator:
    def __init__(self, original_frames_dir,  tolerance, refrac_period, sensor_sensitivity, cropping_limits):

        #self.experiment_timestamp = experiment_timestamp
        self.num_bins = 10
        self.sensor_sensitivity = sensor_sensitivity
        self.cropping_limits = cropping_limits
        self.no_event = 0.5
        self.pos_event = 1.0
        self.neg_event = 0.0
        self.Rw = 0.2126
        self.Gw = 0.7152
        self.Bw = 0.0722
        self.tolerance = tolerance #0.5
        self.refrac_period = refrac_period #0.0001
        #self.timestep = timestep
        self.original_frames_dir = original_frames_dir    ## Path to the directory containing the original frames

        # output directory to save the event frames and event file at the same level as the input directory
        self.base_dir = os.path.dirname(self.original_frames_dir)
        self.events_frames_dir = os.path.join(self.base_dir, "events_frames")
        self.events_file_dir = os.path.join(self.base_dir, "events_file")

    def process_image(self, current_image, previous_image, current_time, last_event_times):
        """
        Processes an entire image for event generation.

        Args:
            current_image: The current frame (HxWx3) as an RGB image.
            previous_image: The previous frame (HxWx3) as an RGB image.
            current_time: The current timestamp for this frame.
            last_event_times: A (HxW) array storing the last event times for each pixel.

        Returns:
            event_image: A (HxW) array with the generated events for each pixel.
            last_event_times: Updated last event times array.
        """
        height = current_image.shape[0]
        width = current_image.shape[1]

        # Convert the images to grayscale using the luminance formula if the image is RGB
        if len(current_image.shape) == 3:
            current_image = self.Rw * current_image[:, :, 0] + self.Gw * current_image[:, :, 1] + self.Bw * current_image[:, :, 2]
            previous_image = self.Rw * previous_image[:, :, 0] + self.Gw * previous_image[:, :, 1] + self.Bw * previous_image[:, :, 2]
        else:
            current_image = current_image
            previous_image = previous_image

        event_image = np.zeros((height, width))
        current_image[current_image == 0] = 1
        previous_image[previous_image == 0] = 1
        log_c = np.log(current_image) * self.sensor_sensitivity
        log_p = np.log(previous_image) * self.sensor_sensitivity

        diff = np.abs(log_c - log_p)
        pol = (log_c >= log_p) * 2 - 1

        max_num_events = (current_time - last_event_times) // self.refrac_period
        num_events = np.minimum(max_num_events, diff // self.tolerance)
        num_events *= pol
        last_event_times[num_events != 0] = current_time

        event_image = np.full(num_events.shape, self.no_event) + (self.no_event / self.num_bins) * num_events

        return event_image, last_event_times
