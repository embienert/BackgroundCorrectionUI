__version__ = "0.6.1 alpha"

from multiprocessing import Pool, cpu_count
from tkinter.filedialog import askopenfilenames, askopenfilename, askdirectory
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Union
from tkinter import Tk
import numpy as np
import os

import BackgroundCorrection.algorithm as algorithm
import BackgroundCorrection.reader as reader
import settings
from BackgroundCorrection import jar
from BackgroundCorrection.parallel import process_parallel
from BackgroundCorrection.roi_integration import get_area, export_rois
from BackgroundCorrection import roi_integration
from BackgroundCorrection.units import convert_x, unit_x_str
from BackgroundCorrection.util import apply_limits, normalize_area, normalize_max, ground, normalize_sum, ranges, \
    dictify
from settings import load_settings

from tqdm import tqdm as loading_bar

import warnings
warnings.filterwarnings("ignore")

try:
    matplotlib.use("QtAgg")
except ValueError:
    matplotlib.use("Qt5Agg")

READFILE_TYPES = [
    ("Raman Files", ".txt .raman"),
    ("Spectra Files", ".spc"),
    ("Chi Files", ".xy .dat"),
]


class DataSet:
    def __init__(self, files: List[reader.DataFile], name: str = "dataset"):
        self.files: List[reader.DataFile] = files
        self.dataset_name: str = name

        if self.dataset_name == "dataset":
            self.dataset_name = ".".join(os.path.basename(self.files[0].filename).split(".")[:-1])

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


class ProcessingResult:
    def __init__(self):
        self.y_ranged = np.array([])

        self.y_jar_corrected = np.array([])
        self.y_jar_scaled = np.array([])
        self.jar_scaling_factor = np.nan

        self.y_background_corrected = np.array([])
        self.y_background_baseline = np.array([])

        self.y_grounded = np.array([])
        self.y_normalized = np.array([])

        self.y_result = np.array([])

        self.label = ""


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

        print(f"Read files: [{', '.join([dataFile.file() for dataFile in dataFiles])}]")

    def extend_headers_legacy(self):
        params = {
            "algorithm": algorithm.algorithm_by_index(self.settings.baseline.algorithm).__name__,
            "itermax": self.settings.baseline.itermax,
            "lambda": self.settings.baseline.lam,
            "ratio": self.settings.baseline.ratio
        } if self.settings.baseline.enable else {}

        for file in self.files:
            file.extend_head(__version__, **params)

    def extend_headers(self, dataset: DataSet):
        general_row = {k: v for k, v in {
                "script_version": f"\"{__version__}\"",
                "first_file": f"\"{dataset.files[0].basename}\"" if len(dataset.files) > 1 else None,
                "last_file": f"\"{dataset.files[-1].basename}\"" if len(dataset.files) > 1 else None,
                "file": f"\"{dataset.files[0].basename}\"" if len(dataset.files) == 1 else None,
                "jar_file": f"\"{dataset.jar_file.basename}\"" if dataset.jar_file else None
            }.items() if v is not None
        }
        baseline_row = {
            "baseline_algorithm": f"\"{algorithm.algorithm_by_index(self.settings.baseline.algorithm).__name__}\"",
            "baseline_itermax": self.settings.baseline.itermax,
            "baseline_lambda": self.settings.baseline.lam,
            "baseline_ratio": self.settings.baseline.ratio
        }
        jar_row = {
            "jar_range_start": self.settings.jar.range_start,
            "jar_range_stop": self.settings.jar.range_stop
        }
        roi_row = {
            "roi_ranges": list(map(lambda x: x[:2], self.settings.rois.ranges)),
            "roi_normalization": f'\"{",".join(np.array(["sum", "sum_linear", "max"])[[value for key, value in self.settings.rois.normalize.items()]])}\"',
            "roi_selection": f'\"{",".join(np.array(["area", "max"])[[value for key, value in self.settings.rois.selection.items()]])}\"'
        }
        normalization_row = {
            "normalization": f'\"{",".join(np.array(["area", "sum", "max"])[[value for key, value in self.settings.normalization.items()][:3]])}\"',
            "ground": self.settings.normalization.ground
        }

        if roi_row["roi_normalization"] == '""':
            roi_row["roi_normalization"] = "none"
        if normalization_row["normalization"] == '""':
            normalization_row["normalization"] = "none"

        header_extension = list(filter(lambda x: x is not None, [
            ", ".join(f"{key}={value}" for key, value in general_row.items()),
            ", ".join(f"{key}={value}" for key, value in
                      baseline_row.items()) if self.settings.io.header.baseline_params and self.settings.baseline.enable else None,
            ", ".join(f"{key}={value}" for key, value in
                      jar_row.items()) if self.settings.io.header.jar_params and self.settings.jar.enable else None,
            ", ".join(f"{key}={value}" for key, value in
                      roi_row.items()) if self.settings.io.header.roi_params and self.settings.rois.enable else None,
            ", ".join(f"{key}={value}" for key, value in
                      normalization_row.items()) if self.settings.io.header.normalization else None,
            ", ".join(settings.option_to_str(self.settings, key) for key in self.settings.io.header.additional) if len(self.settings.io.header.additional) > 0 else None
        ]))


        for file in self.files:
            file.extend_head(header_extension)



    def process(self, intensity, dataset, jar_file=None):
        result = ProcessingResult()

        # Apply x-limits to intensity values
        intensity_ranged, _ = apply_limits(intensity, selection=dataset.range_selection)
        result.y_ranged = intensity_ranged

        # Apply jar-correction to intensity
        intensity_pre_bkg = intensity_ranged
        if self.settings.jar.enable:
            intensity_jar_corrected, jar_intensity_scaled, jar_scaling_factor = jar.jar_correct(jar_file,
                                                                                                intensity_ranged,
                                                                                                **self.bkg_params)

            result.y_jar_corrected = intensity_jar_corrected
            result.y_jar_scaled = jar_intensity_scaled
            result.jar_scaling_factor = jar_scaling_factor

            intensity_pre_bkg = intensity_jar_corrected

        # Perform background correction on prepared intensity data
        intensity_pre_ground = intensity_pre_bkg
        if self.settings.baseline.enable:
            intensity_corrected, baseline = algorithm.correct(intensity_pre_bkg, **self.bkg_params)
            result.y_background_corrected = intensity_corrected
            result.y_background_baseline = baseline

            intensity_pre_ground = intensity_corrected

        # Ground processed intensity data
        intensity_pre_norm = intensity_pre_ground
        if self.settings.normalization.ground:
            intensity_grounded = ground(intensity_pre_ground)
            result.y_grounded = intensity_grounded

            intensity_pre_norm = intensity_grounded

        # Normalize processed intensity data
        intensity_final = intensity_pre_norm
        if self.settings.normalization.area:
            intensity_normalized = normalize_area(x=dataset.x_ranged, y=intensity_pre_norm)
            result.y_normalized = intensity_normalized

            intensity_final = intensity_normalized
        elif self.settings.normalization.sum:
            intensity_normalized = normalize_sum(intensity_pre_norm)
            result.y_normalized = intensity_normalized

            intensity_final = intensity_normalized
        elif self.settings.normalization.max:
            intensity_normalized = normalize_max(intensity_pre_norm)
            result.y_normalized = intensity_normalized

            intensity_final = intensity_normalized

        result.y_result = intensity_final

        return result

    def process_dataset(self, dataset: DataSet):
        print("Processing new dataset")

        # Check if x-Axes match through input files
        print("Checking x-Axes")
        xs = [file.x for file in dataset.files]
        xs_diff = np.diff(np.vstack(xs).reshape(len(xs), -1),
                          axis=0) == 0
        if not xs_diff.all():
            failed_positions = []
            first_x = xs[0]

            for index, x in enumerate(xs[1:]):
                diff = (first_x - x) == 0
                if diff.all():
                    continue

                trouble_indices = np.where(diff == False)[0]
                trouble_indices_ranges = list(ranges(trouble_indices.tolist()))

                trouble_indices_ranges_str = []
                for ti_range in trouble_indices_ranges:
                    start, stop = ti_range
                    if start == stop:
                        trouble_indices_ranges_str.append(str(start))
                    else:
                        trouble_indices_ranges_str.append(str(start) + " to " + str(stop))

                print(f"Found mismatching x-Axis values in file {os.path.basename(dataset.files[index+1].filename)} at indices\n\t"
                      f"{','.join(trouble_indices_ranges_str)}")
                failed_positions.extend(list(zip([index+1 for _ in range(len(trouble_indices_ranges))], trouble_indices_ranges)))

            raise AssertionError(f"x-Axis value mismatch detected in {len(failed_positions)} file(s). Observe program output for exact position.")


        # Convert x-Axis between units and apply limits specified in settings
        x_unit_applied = np.vectorize(convert_x, otypes=[float])(dataset.files[0].x, self.settings.data.unit_in,
                                                                 self.settings.data.unit_out)
        dataset.x_ranged, dataset.range_selection = apply_limits(x_unit_applied,
                                                                 selection_range=(self.settings.data.range_start,
                                                                                  self.settings.data.range_stop))

        # TODO: Option to use same jar for every dataset
        # Prepare jar-correction
        jar_file = None
        if self.settings.jar.enable:
            print("Loading reference file")
            root = Tk()
            filename = askopenfilename(filetypes=READFILE_TYPES, title="Select reference file")
            root.destroy()

            jar_file = jar.load_jar(filename, self.settings.jar.reference_file_head_row_count,
                                    (self.settings.jar.range_start, self.settings.jar.range_stop),
                                    dataset.range_selection)
            dataset.jar_file = jar_file

            print(f"Loaded jar file: {jar_file.file()}")


        dataset_labels = []
        ys = []
        for file in dataset.files:
            dataset_labels.extend(file.labels())

            ys = np.vstack([file.ys for file in dataset.files])

        if self.settings.parallel.enable:
            nr_cores = int(self.settings.parallel.cores) if self.settings.parallel.cores != "auto" else cpu_count()-1

            print("Processing data...")

            with Pool(nr_cores) as pool:
                try:
                    # args = [(intensity, dataset.x_ranged, dataset.range_selection, self.settings, self.bkg_params) for intensity in ys]
                    settings_dict = dictify(self.settings)
                    args = [
                        (
                            intensity,
                            dataset.x_ranged,
                            dataset.range_selection,
                            jar_file, settings_dict,
                            self.bkg_params,
                            label,
                            index in self.settings.baseline.plot.test_datasets
                        ) for index, (intensity, label) in enumerate(zip(ys, dataset_labels))
                    ]

                    size = len(ys)
                    results = pool.starmap(process_parallel, loading_bar(args, total=size))
                except KeyboardInterrupt:
                    print("CTRL+C was pressed. Killing all processes.")

                    pool.terminate()
                    raise KeyboardInterrupt
        else:
            results = []
            for column_index, (intensity, label) in enumerate(zip(ys, dataset_labels)):
                print(f"Processing column {column_index} of file {os.path.basename(file.filename)}")
                result = self.process(intensity, dataset, jar_file)
                result.label = label

                # Plot data for first sample in first file
                if column_index in self.settings.baseline.plot.test_datasets:
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
                            plt.plot(dataset.x_ranged, result.y_jar_scaled, label="Jar Intensity (Corrected, Scaled)")
                        if self.settings.jar.plot.intensity_original:
                            plt.plot(dataset.x_ranged, result.y_ranged, label="Intensity (Pre-Jar-Correction)")
                        if self.settings.jar.plot.intensity_corrected:
                            plt.plot(dataset.x_ranged, result.y_jar_corrected, label="Intensity (Jar-Corrected)")

                        plt.xlabel("x")
                        plt.ylabel("intensity")
                        plt.legend(loc="upper right")
                        plt.title(label + " (jar)")

                        plt.show()

                    if self.settings.baseline.plot.enable and self.settings.baseline.enable:
                        intensity_ranged_plot = result.y_ranged if not self.settings.jar.enable else result.y_jar_corrected
                        intensity_original_label = "Intensity (Original)" if not self.settings.jar.enable else "Intensity (Jar-Corrected)"

                        if self.settings.baseline.plot.original:
                            plt.plot(dataset.x_ranged, intensity_ranged_plot, label=intensity_original_label)
                        if self.settings.baseline.plot.baseline:
                            plt.plot(dataset.x_ranged, result.y_background_baseline, label="Baseline")
                        if self.settings.baseline.plot.corrected:
                            plt.plot(dataset.x_ranged, result.y_background_corrected, label="Intensity (Corrected)")
                        if self.settings.baseline.plot.corrected_normalized:
                            plt.plot(dataset.x_ranged, result.y_result, label="Intensity (Corrected, Normalized")

                        plt.xlabel("x")
                        plt.ylabel("intensity")
                        plt.legend(loc="upper right")
                        plt.title(label)

                        plt.show()

                results.append(result)

        # Join results of individual files into single array
        dataset.x_result = dataset.x_ranged
        dataset.ys_result = np.vstack([result.y_result for result in results])
        dataset.ys_jar = np.vstack([result.y_jar_corrected for result in results])

        base_dir = dataset.base_dir() if not os.path.isabs(self.settings.io.out_dir) else ""
        baseline_out_dir = os.path.join(base_dir, self.settings.io.out_dir, self.settings.baseline.out_dir)

        if not os.path.exists(baseline_out_dir):
            os.mkdir(baseline_out_dir)

        self.extend_headers(dataset)

        # Write final results to output file
        if self.settings.baseline.enable:
            dataset.export_baseline(baseline_out_dir, sep=self.settings.io.dat_file_sep)

        # Write jar-corrected data to output file
        if self.settings.jar.enable:
            jar_out_dir = os.path.join(base_dir, self.settings.io.out_dir, self.settings.jar.out_dir)
            dataset.export_jar(jar_out_dir, sep=self.settings.io.dat_file_sep)

        # Plotting
        if self.settings.plot.enable or (self.settings.rois.enable and self.settings.rois.plot.enable):
            # Generate subplots
            if self.settings.rois.enable and self.settings.rois.plot.enable:
                fig, (ax_intensity, ax_rois) = plt.subplots(1, 2, sharey="row",
                                                            gridspec_kw={"width_ratios": self.settings.rois.plot.ratio})
            else:
                fig, ax_intensity = plt.subplots()

            # Plot result intensities
            y_scale = np.arange(0, dataset.ys_result.shape[0] * self.settings.plot.time_step,
                                self.settings.plot.time_step)[:dataset.ys_result.shape[0]]

            time_range_selection = (self.settings.plot.display_time_range_start <= y_scale) & (y_scale <= self.settings.plot.display_time_range_stop)
            y_scale = y_scale[time_range_selection]

            print(dataset.x_result.shape, y_scale.shape)

            extent = [np.min(dataset.x_result), np.max(dataset.x_result), np.min(y_scale), np.max(y_scale)]

            if not self.settings.plot.flip_y_data:
                # img_data = np.flip(dataset.ys_result, axis=0)
                img_data = np.flip(dataset.ys_result, axis=0)[time_range_selection]
            else:
                # img_data = dataset.ys_result
                img_data = dataset.ys_result[time_range_selection[::-1]]

            if self.settings.plot.flip_x_data:
                img_data = np.flip(img_data, axis=1)

            if self.settings.plot.flip_y_ticks:
                extent[2], extent[3] = extent[3], extent[2]

            if self.settings.plot.flip_x_ticks:
                extent[0], extent[1] = extent[1], extent[0]

            # Plot intensity data as heatmap and scatter ROI integration areas as
            intensity_plot = ax_intensity.imshow(img_data, extent=extent, cmap=self.settings.plot.heatmap)

            # Set plot options
            if self.settings.plot.colorbar:
                fig.colorbar(intensity_plot, ax=ax_intensity)
            ax_intensity.set_xlabel(
                unit_x_str(self.settings.data.unit_out) if not self.settings.plot.x_unit else self.settings.plot.x_unit)
            ax_intensity.set_ylabel(f"Time [{self.settings.plot.y_unit}]")
            ax_intensity.set_xlim(extent[0], extent[1])
            ax_intensity.set_ylim(extent[2], extent[3])
            ax_intensity.set_aspect("auto")

        # ROI integration data processing
        if self.settings.rois.enable:
            if dataset.x_result[0] > dataset.x_result[-1]:
                # x-Axis decrementing --> Flip ROIs for successful integration
                flip_roi = True
            else:
                flip_roi = False

            # Get ROI integration data on whole DataSet intensity
            select_max = self.settings.rois.selection.max
            dataset.roi_values = np.array(
                [[get_area(dataset.x_result, y_result, range_min, range_max, flip=flip_roi, select_max=select_max) for y_result in
                  dataset.ys_result]
                 for (range_min, range_max, _) in self.settings.rois.ranges]
            )

            if self.settings.rois.normalize.sum:
                dataset.roi_values_normalized, mean_error = roi_integration.normalize(dataset.roi_values)
            elif self.settings.rois.normalize.sum_linear:
                dataset.roi_values_normalized, mean_error = roi_integration.normalize_linear(dataset.roi_values)
            elif self.settings.rois.normalize.max:
                dataset.roi_values_normalized, mean_error = roi_integration.normalize_max(dataset.roi_values)
            else:
                dataset.roi_values_normalized, mean_error = dataset.roi_values, 0

            # Write ROI integration data to output file
            rois_out_dir = os.path.join(base_dir, self.settings.io.out_dir, self.settings.rois.out_dir)
            export_rois(dataset.roi_values_normalized, dataset_labels,
                        self.settings.rois.ranges, rois_out_dir, dataset.dataset_name,
                        time_step=self.settings.plot.time_step, time_unit=self.settings.plot.y_unit)

            # Plot ROI integration data
            if self.settings.rois.plot.enable:
                roi_values_normalized_ranged = dataset.roi_values_normalized[:, time_range_selection]

                if self.settings.plot.flip_y_data:
                    rois_data = np.flip(roi_values_normalized_ranged, axis=1)
                else:
                    rois_data = roi_values_normalized_ranged

                if self.settings.plot.flip_y_ticks:
                    y_scale = np.flip(y_scale)

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

        if self.settings.plot.enable:
            if not (self.settings.rois.enable and self.settings.rois.plot.enable):
                fig.savefig(os.path.join(baseline_out_dir, dataset.dataset_name + ".png"))
            # fig.show()
            ax = ax_intensity

            plt.close(fig)

            # Create interactive plot
            plt.sca(ax)
            plt.show()

    def run(self):
        # Load files and extend headers with parameters specified in settings
        self.load_files()

        dataset = DataSet(self.files)

        self.process_dataset(dataset)


if __name__ == "__main__":
    controller = Controller()
    controller.run()
