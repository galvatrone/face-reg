import argparse
import os

import cv2
import numpy as np

from common import (
    TOOLS_DIR,
    compare_encodings,
    ensure_input_dir,
    get_first_two_valid_faces,
    load_first_face_encoding,
    resolve_input_path,
)

DEFAULT_OUTPUT = os.path.join(TOOLS_DIR, "face_compare_result.jpg")


def resize_to_height(image, target_height):
    if image.shape[0] == target_height:
        return image

    scale = target_height / image.shape[0]
    target_width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_face_box(image, location, label):
    top, right, bottom, left = location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 200, 0), 2)
    text_y = max(30, top - 10)
    cv2.putText(
        image,
        label,
        (left, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 0),
        2,
        cv2.LINE_AA,
    )


def build_preview(
    image_path_1,
    image_path_2,
    location_1,
    location_2,
    similarity_percent,
    is_match,
    output_path,
):
    image_1 = cv2.imread(image_path_1)
    image_2 = cv2.imread(image_path_2)
    if image_1 is None:
        raise ValueError(f"Failed to open image: {image_path_1}")
    if image_2 is None:
        raise ValueError(f"Failed to open image: {image_path_2}")

    draw_face_box(image_1, location_1, "Photo 1")
    draw_face_box(image_2, location_2, "Photo 2")

    target_height = max(image_1.shape[0], image_2.shape[0])
    image_1 = resize_to_height(image_1, target_height)
    image_2 = resize_to_height(image_2, target_height)

    spacer = 20
    canvas_width = image_1.shape[1] + image_2.shape[1] + spacer
    header_height = 80
    canvas = np.full((header_height + target_height, canvas_width, 3), 255, dtype=np.uint8)

    canvas[header_height : header_height + target_height, 0 : image_1.shape[1]] = image_1
    start_x = image_1.shape[1] + spacer
    canvas[header_height : header_height + target_height, start_x : start_x + image_2.shape[1]] = image_2

    match_text = "MATCH" if is_match else "NO MATCH"
    title = f"Similarity: {similarity_percent:.2f}% | {match_text}"
    cv2.putText(
        canvas,
        title,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(output_path, canvas)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two photos by face similarity and save a side-by-side preview."
    )
    parser.add_argument("image_1", nargs="?", help="Path to the first photo")
    parser.add_argument("image_2", nargs="?", help="Path to the second photo")
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output preview path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = ensure_input_dir()
    skipped_files = []
    if args.image_1 and args.image_2:
        image_path_1 = resolve_input_path(args.image_1)
        image_path_2 = resolve_input_path(args.image_2)
        encoding_1, location_1 = load_first_face_encoding(image_path_1)
        encoding_2, location_2 = load_first_face_encoding(image_path_2)
    elif args.image_1 or args.image_2:
        raise ValueError("Provide either both image paths or no paths at all.")
    else:
        face_1, face_2, skipped_files = get_first_two_valid_faces()
        image_path_1, encoding_1, location_1 = face_1
        image_path_2, encoding_2, location_2 = face_2

    distance, similarity_percent, is_match = compare_encodings(encoding_1, encoding_2)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    build_preview(
        image_path_1,
        image_path_2,
        location_1,
        location_2,
        similarity_percent,
        is_match,
        args.output,
    )

    print(f"Input folder: {input_dir}")
    print(f"Photo 1: {os.path.abspath(image_path_1)}")
    print(f"Photo 2: {os.path.abspath(image_path_2)}")
    if skipped_files:
        print("Skipped files without detectable face:")
        for skipped_path in skipped_files:
            print(f"  {os.path.abspath(skipped_path)}")
    print(f"Face distance: {distance:.4f}")
    print(f"Similarity: {similarity_percent:.2f}%")
    print(f"Match: {'YES' if is_match else 'NO'}")
    print(f"Saved preview: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
