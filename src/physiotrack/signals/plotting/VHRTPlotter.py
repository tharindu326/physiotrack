##################################################
### Real time data plotter in opencv (python)  ###
## Plot integer data for debugging and analysis ##
##################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io


# Plot values in opencv program
class RT2DPlotter:
    def __init__(self, plot_width, plot_height, max_values_y_axis, seconds_x_axis=10, number_samples_x_axis=150, fps=30,
                 number_of_signals=1, plotting_method="MPL", title='', axisLabels=None, axisLimits=None, signalLabels=None,
                 colors=None, figure=None, canvas=None, index=0, dpi=100):

        # Plotting methods:
        #    - MPL: MatPlotLib
        #    - OCV: OpenCV

        # Variables initialization
        self.width = plot_width
        self.height = plot_height
        self.color = (255, 0, 0)
        self.max_y_value = 0
        self.min_y_value = 0
        self.fps = fps
        self.dpi = dpi
        self.number_of_seconds_x_axis = int(number_samples_x_axis / fps)
        self.samples_number_x_axis = number_samples_x_axis  # Default is assuming 10 seconds and 30 fps
        if colors is not None:
            self.colors = colors
        else:
            self.colors = ['red', 'green', 'blue', 'orange', 'brown', 'black']

        if axisLabels is not None:
            self.axisLabels = axisLabels
        else:
            self.axisLabels = ['Seconds (s)', 'Normalized Acceleration (m/s)']

        if axisLimits is not None:
            self.axisLimits = axisLimits
        else:
            self.axisLimits = [0, self.number_of_seconds_x_axis, -1.0, 1.0]

        if signalLabels is not None:
            self.signalLabels = signalLabels
        else:
            self.signalLabels = ['acc x coord', 'acc y coord', 'acc z coord']
        self.plotting_method = plotting_method
        self.max_values_y_axis = max_values_y_axis  # Expect array with the same number of maximums than input signals
        self.number_of_signals = number_of_signals
        self.values_arrays = {}
        self.title = title
        self.fig = []
        self.canvas = []
        self.buf = []
        self.io_buf = []
        self.index = index
        # plt.ion()  # Note this correction

        for i in range(number_of_signals):
            self.values_arrays[str('values_signal_{i}'.format(i=i))] = []
        if plotting_method == "MPL":
            self.plot_image = []
            for i in range(number_of_signals):
                for j in range(self.samples_number_x_axis):
                    self.values_arrays[str('values_signal_{i}'.format(i=i))].append(0)

            self.x = np.linspace(0, self.number_of_seconds_x_axis, self.samples_number_x_axis)
            if canvas is None:
                print("          >> [RT2DPlotter] Creating new figure with id = ", self.index)
                self.fig = plt.figure(self.index, figsize=(self.width/self.dpi, self.height/self.dpi), dpi=self.dpi)
                self.canvas = FigureCanvasAgg(self.fig)
            else:
                self.fig = figure
                self.canvas = canvas
        else:
            self.plot_image = np.ones((self.height, self.width, 3)) * 255  # For OpenCV option

    def plot(self, values, label='plot'):
        # Update new values in plot
        # values: scalar or array with data for plotting in Real-Time.
        # values contains the samples of signals in the current "frame".

        # isinstance(P, (list, tuple, np.ndarray))
        if self.plotting_method == "MPL":
            if self.number_of_signals == 1:
                values = values / self.max_values_y_axis
            else:
                for i in range(self.number_of_signals):
                    values[i] = values[i] / self.max_values_y_axis[i]
        elif self.plotting_method == "OCV":
            values = int(values)

        # print( "            >>>> Valores normalizados: " + str(self.index) + " - " + str(values) + "\n")
        # self.max_y_value = max(values)
        # self.min_y_value = min(values)
        if self.number_of_signals == 1:
            self.values_arrays[str('values_signal_{i}'.format(i=0))].append(values)
        else:
            for i in range(self.number_of_signals):
                self.values_arrays[str('values_signal_{i}'.format(i=i))].append(values[i])

        if self.plotting_method == "OCV":
            while len(self.values_arrays['values_signal_0']) > self.width:
                self.val.pop(0)
        elif self.plotting_method == "MPL":
            while len(self.values_arrays['values_signal_0']) > len(self.x):
                for i in range(self.number_of_signals):
                    self.values_arrays[str('values_signal_{i}'.format(i=i))].pop(0)
                # self.val.pop(0)

        self.show_plot(label)
        return self.plot_image

    def show_plot(self, label):
        # Show plot using opencv imshow
        if self.plotting_method == "OCV":
            self.plot_image = np.ones((self.height, self.width, 3)) * 255
            cv2.line(self.plot_image, (0, self.height // 2), (self.width, self.height // 2), (0, 255, 0), 1)
            for i in range(len(self.val) - 1):
                cv2.line(self.plot_image, (i, self.height // 2 - self.val[i]), (i + 1, self.height // 2 - self.val[i + 1]), self.color, 1)

            cv2.imshow(label, self.plot_image)
            cv2.waitKey(2)
        elif self.plotting_method == "MPL":
            self.fig = plt.figure(self.index, figsize=(self.width/self.dpi, self.height/self.dpi), dpi=self.dpi)
            plt.clf()
            plt.axis(self.axisLimits)
            plt.ylim(self.axisLimits[2], self.axisLimits[3])

            for i in range(self.number_of_signals):
                vector_values = np.copy(self.values_arrays[str('values_signal_{i}'.format(i=i))])
                vector_values[vector_values == 0] = np.nan
                # mean = np.nanmean(vector_values)
                # if self.number_of_signals == 1:
                #     print("===============================")
                #     print(self.x, self.values_arrays[str('values_signal_{i}'.format(i=i))])
                plt.plot(self.x, self.values_arrays[str('values_signal_{i}'.format(i=i))], color=self.colors[i], label=self.signalLabels[i], linewidth=1.1)

            # All this settings could be made in the initialization
            plt.yticks(fontsize=5)
            plt.grid()
            self.fig.suptitle(self.title, fontsize=6)
            plt.xlabel(self.axisLabels[0], fontsize=5)
            plt.ylabel(self.axisLabels[1], fontsize=5)
            plt.subplots_adjust(bottom=.25, left=.125)
            plt.legend(loc='lower left', prop={'size': 5})

            # Option 1: Return image.
            # self.plot_image = self.get_img_from_fig()

            # Option 2: Return a numpy array using IO package.
            # self.io_buf = io.BytesIO()
            # self.fig.savefig(self.io_buf, format='rgba', dpi=100)
            # self.io_buf.seek(0)
            # self.plot_image = np.reshape(np.frombuffer(self.io_buf.getvalue(), dtype=np.uint8),
            # 				 newshape=(int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1))
            # self.io_buf.close()

            # Option 3_ Return a numpy array from Canvas.
            self.canvas.draw()
            self.buf = self.canvas.buffer_rgba()
            self.plot_image = np.asarray(self.buf)

    # define a function which returns an image as numpy array from figure
    def get_img_from_fig(self, dpi=180):
        self.buf = io.BytesIO()
        self.fig.savefig(self.buf, format="png", dpi=dpi)
        self.buf.seek(0)
        img_arr = np.frombuffer(self.buf.getvalue(), dtype=np.uint8)
        self.buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imshow(self.title, img)

        return img
