from model.model import LiIonModel

# Wire the concrete model for the runner. The runner imports `handler`
# from here and calls handler.process(mode, values, files, output_dir, ...).
handler = LiIonModel()