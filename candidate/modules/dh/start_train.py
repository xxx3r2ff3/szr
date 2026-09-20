import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to video file")
    parser.add_argument("--bitrate", type=str, default="3000k")
    parser.add_argument("--no_train", action="store_true")
    args, unknow = parser.parse_known_args()

    from train import main as train
    from process import main as process
    from network import export_onnx
    from onnx import load_model, save_model
    from onnxmltools.utils import float16_converter

    process(args.path, bitrate=args.bitrate)
    if not args.no_train:
        train(args.path)

        # 导出onnx
        pth_dir = os.path.join(os.path.dirname(args.path), "checkpoint")
        files = os.listdir(pth_dir)
        if not len(files):
            raise Exception("Not found checkpoint")

        best_file = files[0]
        best_file = os.path.join(pth_dir, best_file)
        print(best_file)
        export_onnx(best_file)

        # 导出fp16 onnx
        onnx_file = os.path.join(pth_dir, "model.onnx")
        onnx_model = load_model(onnx_file)
        trans_model = float16_converter.convert_float_to_float16(
            onnx_model, keep_io_types=True
        )
        output = onnx_file.replace(".onnx", "_fp16.onnx")
        save_model(trans_model, output)
