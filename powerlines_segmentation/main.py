from powerlines_segmentation.trainer import Trainer
from powerlines_segmentation.http import start_http_server
from powerlines_segmentation.segmentor import Segmentor
import argparse


def train():
    trainer = Trainer()
    trainer.train()


def segment(image_path: str):
    segmentor = Segmentor()
    segmentor.segment(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train and use powerlines segmentation"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models",
    )
    parser.add_argument(
        "--segment",
        help="Segments a provided image",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Starts a HTTP inference server",
    )
    args = parser.parse_args()

    if args.train:
        train()

    if args.http:
        start_http_server()

    if args.segment:
        segment(args.segment)


if __name__ == "__main__":
    main()
