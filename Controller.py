__version__ = "0.0 alpha"

from tkinter.filedialog import askopenfilenames, askopenfilename, askdirectory
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Union
from tkinter import Tk
import numpy as np
import os

import BackgroundCorrection.algorithm as algorithm
import BackgroundCorrection.reader as reader
from BackgroundCorrection import jar
from BackgroundCorrection.roi_integration import get_area, normalize, export_rois
from BackgroundCorrection.units import convert_x, unit_x_str
from BackgroundCorrection.util import apply_limits, normalize_area, normalize_max, ground, normalize_sum
from settings import load_settings

matplotlib.use("QtAgg")

READFILE_TYPES = [
    ("Raman Files", ".txt .raman"),
    ("Spectra Files", ".spc"),
    ("Chi Files", ".xy .dat"),
]


class DataSet:
    def __init__(self, files: List[reader.DataFile], name: str = "dataset"):
        self.files: List[reader.DataFile] = files
        self.dataset_name: str = name

        self.jar_file: Union[reader.DataFile, None] = None

        self.x_ranged = None
        self.range_selection = None

        self.x_result: np.ndarray = np.array([])
        self.ys_result: np.ndarray = np.array([])
        self.ys_jar: np.ndarray = np.array([])

    def export_baseline(self, out_dir, sep):
        baseline_export_file = reader.DataFile(filename=self.dataset_name + ".dat", content=np.array([]),
                                               head=self.files[0].head)

        baseline_export_file.x_result = self.x_result
        baseline_export_file.ys_result = self.ys_result

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        baseline_export_file.write_dat(out_dir, sep)

    def export_jar(self, out_dir, sep):
        jar_export_file = reader.DataFile(filename=self.dataset_name + "_jar.dat", content=np.array([]), head=[
            f"Jar-Corrected intensities with reference file {self.jar_file.filename}"])

        jar_export_file.x_result = self.x_result
        jar_export_file.ys_result = self.ys_jar

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        jar_export_file.write_dat(out_dir, sep)

    def base_dir(self):
        return os.path.dirname(os.path.abspath(self.files[0].filename))


class Controller:
    def __init__(self):
        # TODO: Load settings / set defaults
        self.settings = load_settings()
        self.bkg_params = {
            "fkt": algorithm.algorithm_by_index(self.settings.baseline.algorithm),
            "lam": self.settings.baseline.lam,
            "ratio": self.settings.baseline.ratio,
            "itermax": self.settings.baseline.itermax
        }

        self.files: List[reader.DataFile] = []

    def load_files(self):
        root = Tk()
        files = askopenfilenames(filetypes=READFILE_TYPES, title="Select files to process")
        root.destroy()

        if not files:
            files = []

            root = Tk()
            files_dir = askdirectory()
            root.destroy()

            if files_dir == "":
                raise FileNotFoundError("No path provided. Aborting.")

            files_subdirs = [subdir.path for subdir in os.scandir(files_dir)
                             if subdir.is_dir() and subdir.name != os.path.basename(self.settings.io.out_dir)]
            subdir_files = [[file.path for file in os.scandir(subdir) if os.path.isfile(file)]
                            for subdir in [files_dir, *files_subdirs]]

            for file_list in subdir_files:
                files.extend(file_list)

        # TODO: Log successful file selection

        dataFiles = reader.read_many(files, self.settings.io.head_row_count)
        self.files = dataFiles

    def extend_headers(self):
        params = {
            "algorithm": algorithm.algorithm_by_index(self.settings.baseline.algorithm).__name__,
            "itermax": self.settings.baseline.itermax,
            "lambda": self.settings.baseline.lam,
            "ratio": self.settings.baseline.ratio
        } if self.settings.baseline.enable else {}

        for file in self.files:
            file.extend_head(__version__, **params)

    def process_dataset(self, dataset: DataSet):
        print("Processing new dataset")

        # Check if x-Axes match through input files
        print("Checking x-Axes")
        xs = [file.x for file in dataset.files]
        assert (np.diff(np.vstack(xs).reshape(len(xs), -1),
                        axis=0) == 0).all(), "x-Axes through read files do not match"

        # Convert x-Axis between units and apply limits specified in settings
        x_unit_applied = np.vectorize(convert_x, otypes=[float])(dataset.files[0].x, self.settings.data.unit_in,
                                                                    self.settings.data.unit_out)
        dataset.x_ranged, dataset.range_selection = apply_limits(x_unit_applied,
                                                                 selection_range=(self.settings.data.range_start,
                                                                                  self.settings.data.range_stop))

        # TODO: Option to use same jar for every dataset
        # Prepare jar-correction
        if self.settings.jar.enable:
            print("Loading reference file")
            root = Tk()
            filename = askopenfilename(filetypes=READFILE_TYPES, title="Select reference file")
            root.destroy()

            jar_file = jar.load_jar(filename, self.settings.jar.reference_file_head_row_count,
                                    (self.settings.jar.range_start, self.settings.jar.range_stop),
                                    dataset.range_selection)
            dataset.jar_file = jar_file

        # Process data
        for file_index, file in enumerate(dataset.files):
            print(f"Processing file {file_index}")

            ys_ranged = []
            ys_jar_corrected = []
            ys_jar_scaled = []
            jar_scaling_factors = []
            ys_background_corrected = []
            ys_background_baseline = []
            ys_normalized = []
            ys_grounded = []
            ys_result = []

            file.x_ranged = dataset.x_ranged

            for column_index, intensity in enumerate(file.ys):
                print(f"Processing column {column_index} of file {file_index}")
                # Apply x-limits to intensity values
                intensity_ranged, _ = apply_limits(intensity, selection=dataset.range_selection)
                ys_ranged.append(intensity_ranged)

                # Apply jar-correction to intensity
                intensity_pre_bkg = intensity_ranged
                if self.settings.jar.enable:
                    intensity_jar_corrected, jar_intensity_scaled, jar_scaling_factor = jar.jar_correct(jar_file,
                                                                                                        intensity_ranged,
                                                                                                        **self.bkg_params)

                    ys_jar_corrected.append(intensity_jar_corrected)
                    ys_jar_scaled.append(jar_intensity_scaled)
                    jar_scaling_factors.append(jar_scaling_factor)

                    intensity_pre_bkg = intensity_jar_corrected

                # Perform background correction on prepared intensity data
                intensity_pre_ground = intensity_pre_bkg
                baseline = []
                if self.settings.baseline.enable:
                    intensity_corrected, baseline = algorithm.correct(intensity_pre_bkg, **self.bkg_params)
                    ys_background_corrected.append(intensity_corrected)
                    ys_background_baseline.append(baseline)

                    intensity_pre_ground = intensity_corrected

                # Ground processed intensity data
                intensity_pre_norm = intensity_pre_ground
                if self.settings.normalization.ground:
                    intensity_grounded = ground(intensity_pre_ground)
                    ys_grounded.append(intensity_grounded)

                    intensity_pre_norm = intensity_grounded

                # Normalize processed intensity data
                intensity_final = intensity_pre_norm
                if self.settings.normalization.area:
                    intensity_normalized = normalize_area(x=dataset.x_ranged, y=intensity_pre_norm)
                    ys_normalized.append(intensity_normalized)

                    intensity_final = intensity_normalized
                elif self.settings.normalization.sum:
                    intensity_normalized = normalize_sum(intensity_pre_norm)
                    ys_normalized.append(intensity_normalized)

                    intensity_final = intensity_normalized
                elif self.settings.normalization.max:
                    intensity_normalized = normalize_max(intensity_pre_norm)
                    ys_normalized.append(intensity_normalized)

                    intensity_final = intensity_normalized

                ys_result.append(intensity_final)

                # Plot data for first sample in first file
                if file_index == column_index == 0:
                    if self.settings.jar.plot.enable and self.settings.jar.enable:
                        if self.settings.jar.plot.jar_original:
                            plt.plot(dataset.x_ranged, jar_file.ys[0], label="Jar Intensity (Original)")
                        if self.settings.jar.plot.jar_ranged:
                            plt.plot(jar_file.x_ranged, jar_file.ys_ranged[0], label="Jar Intensity (Ranged)")
                        if self.settings.jar.plot.jar_baseline:
                            plt.plot(jar_file.x_ranged, jar_file.ys_background_baseline[0],
                                     label="Jar Baseline (Ranged)")
                        if self.settings.jar.plot.jar_corrected:
                            plt.plot(jar_file.x_ranged, jar_file.ys_background_corrected[0],
                                     label="Jar Intensity (Corrected, Ranged)")
                        if self.settings.jar.plot.jar_scaled:
                            plt.plot(dataset.x_ranged, jar_intensity_scaled, label="Jar Intensity (Corrected, Scaled)")
                        if self.settings.jar.plot.intensity_original:
                            plt.plot(dataset.x_ranged, intensity_ranged, label="Intensity (Pre-Jar-Correction)")
                        if self.settings.jar.plot.intensity_corrected:
                            plt.plot(dataset.x_ranged, intensity_jar_corrected, label="Intensity (Jar-Corrected)")

                        plt.xlabel("x")
                        plt.ylabel("intensity")
                        plt.legend(loc="upper right")
                        plt.title(f"file {file_index}, column {column_index}")

                        plt.show()

                    if self.settings.baseline.plot.enable and self.settings.baseline.enable:
                        if self.settings.baseline.plot.original:
                            plt.plot(dataset.x_ranged, intensity_ranged, label="Intensity (Original)")
                        if self.settings.baseline.plot.baseline:
                            plt.plot(dataset.x_ranged, baseline, label="Baseline")
                        if self.settings.baseline.plot.corrected:
                            plt.plot(dataset.x_ranged, intensity_pre_norm, label="Intensity (Corrected)")
                        if self.settings.baseline.plot.corrected_normalized:
                            plt.plot(dataset.x_ranged, intensity_final, label="Intensity (Corrected, Normalized")

                        plt.xlabel("x")
                        plt.ylabel("intensity")
                        plt.legend(loc="upper right")
                        plt.title(f"file {file_index}, column {column_index}")

                        plt.show()

            # Join column data per-file
            file.ys_ranged = np.array(ys_ranged)
            file.ys_jar_corrected = np.array(ys_jar_corrected)
            file.jar_scaling_factors = np.array(jar_scaling_factors)
            file.ys_background_corrected = np.array(ys_background_corrected)
            file.ys_background_baseline = np.array(ys_background_baseline)
            file.ys_grounded = np.array(ys_grounded)
            file.ys_normalized = np.array(ys_normalized)

            file.ys_result = np.array(ys_result)
            file.x_result = dataset.x_ranged

        # Join results of individual files into single array
        dataset.x_result = dataset.files[0].x_result if len(dataset.files) != 0 else None
        dataset.ys_result = np.vstack([file.ys_result for file in dataset.files])
        dataset.ys_jar = np.vstack([file.ys_jar_corrected for file in dataset.files])

        base_dir = dataset.base_dir() if not os.path.isabs(self.settings.io.out_dir) else ""

        # Write final results to output file
        if self.settings.baseline.enable:
            baseline_out_dir = os.path.join(base_dir, self.settings.io.out_dir, self.settings.baseline.out_dir)
            dataset.export_baseline(baseline_out_dir, sep=self.settings.io.dat_file_sep)

        # Write jar-corrected data to output file
        if self.settings.jar.enable:
            jar_out_dir = os.path.join(base_dir, self.settings.io.out_dir, self.settings.jar.out_dir)
            dataset.export_jar(jar_out_dir, sep=self.settings.io.dat_file_sep)


        # Plotting
        if self.settings.baseline.plot.enable or (self.settings.rois.enable and self.settings.rois.plot.enable):
            # Generate subplots
            if self.settings.rois.enable and self.settings.rois.plot.enable:
                fig, (ax_intensity, ax_rois) = plt.subplots(1, 2, sharey="row", gridspec_kw={"width_ratios": [5, 2]})
            else:
                fig, ax_intensity = plt.subplots()

            # Plot result intensities
            y_scale = np.arange(0, dataset.ys_result.shape[0] * self.settings.rois.plot.time_step)
            extent = [np.min(dataset.x_result), np.max(dataset.x_result), np.min(y_scale), np.max(y_scale)]

            if self.settings.rois.plot.flip_y:
                img_data = np.flip(dataset.ys_result)
            else:
                img_data = dataset.ys_result

                ax_intensity.invert_yaxis()

            # Plot intensity data as heatmap and scatter ROI integration areas as
            ax_intensity.imshow(img_data, extent=extent, cmap=self.settings.rois.plot.heatmap)

            # Set plot options
            ax_intensity.set_xlabel(unit_x_str(self.settings.data.unit_out))
            ax_intensity.set_ylabel("Time [s]")
            ax_intensity.set_xlim(extent[0], extent[1])
            ax_intensity.set_ylim(extent[2], extent[3])
            ax_intensity.set_aspect("auto")


        # ROI integration data processing
        if self.settings.rois.enable:
            # Get ROI integration data on whole DataSet intensity
            dataset.roi_values = np.array(
                [[get_area(dataset.x_result, y_result, range_min, range_max) for y_result in dataset.ys_result]
                 for (range_min, range_max, _) in self.settings.rois.ranges]
            )

            dataset.roi_values_normalized, mean_error = normalize(dataset.roi_values)

            # Write ROI integration data to output file
            rois_out_dir = os.path.join(base_dir, self.settings.io.out_dir, self.settings.rois.out_dir)
            export_rois(dataset.roi_values_normalized, [file.filename for file in dataset.files],
                        self.settings.rois.ranges, rois_out_dir, dataset.dataset_name)

            # Plot ROI integration data
            if self.settings.rois.plot.enable:
                if self.settings.rois.plot.flip_y:
                    rois_data = dataset.roi_values_normalized[:, ::-1]
                else:
                    rois_data = dataset.roi_values_normalized

                # Plot intensity data as heatmap and scatter ROI integration areas as
                for roi_areas, (_, _, color) in zip(rois_data, self.settings.rois.ranges):
                    if color.strip() == "":
                        color = None

                    ax_rois.scatter(roi_areas, y_scale, s=5, c=color)

                # Set plot options
                ax_rois.tick_params(
                    axis='y',
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False
                )
                ax_rois.set_xlabel("Norm. Intensity")

                # Save figure to file
                fig.tight_layout()
                fig.savefig(os.path.join(rois_out_dir, dataset.dataset_name + "_rois.png"))

        if self.settings.baseline.plot.enable:
            if not (self.settings.rois.enable and self.settings.rois.plot.enable):
                fig.savefig(os.path.join(baseline_out_dir, dataset.dataset_name + ".png"))
            plt.close(fig)


    def run(self):
        # Load files and extend headers with parameters specified in settings
        self.load_files()
        self.extend_headers()

        dataset = DataSet(self.files)

        self.process_dataset(dataset)


if __name__ == "__main__":
    controller = Controller()
    controller.run()