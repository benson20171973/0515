import numpy as np
import mediapipe as mp
import cv2
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers

IMAGE_FILENAMES = ['f.jpg']
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px
   
x = 0.68 #@param {type:"slider", min:0, max:1, step:0.01}
y = 0.68 #@param {type:"slider", min:0, max:1, step:0.01}


BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# Create the options that will be used for InteractiveSegmenter
base_options = python.BaseOptions(model_asset_path='magic_touch.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,output_category_mask=True)

# Create the interactive segmenter
with vision.InteractiveSegmenter.create_from_options(options) as segmenter:

  # Loop through demo image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe image file that will be segmented
    image = mp.Image.create_from_file(image_file_name)

    # Retrieve the masks for the segmented image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x, y))
    segmentation_result = segmenter.segment(image, roi)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    output_image = np.where(condition, fg_image, bg_image)

    # Draw a white dot with black border to denote the point of interest
    thickness, radius = 6, -1
    keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
    cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
    cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)
    cv2.imshow('My Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f'Segmentation mask of {image_file_name}:')
    

    with python.vision.InteractiveSegmenter.create_from_options(options) as segmenter:
      for image_file_name in IMAGE_FILENAMES:

        # Create the MediaPipe Image
        image = mp.Image.create_from_file(image_file_name)

        # Retrieve the category masks for the image
        roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                            keypoint=NormalizedKeypoint(x, y))
        segmentation_result = segmenter.segment(image, roi)
        category_mask = segmentation_result.category_mask

        # Convert the BGR image to RGB
        image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

        # Apply effects
        blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        output_image = np.where(condition, image_data, blurred_image)

        # Draw a white dot with black border to denote the point of interest
        thickness, radius = 6, -1
        keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
        cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
        cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)
        cv2.imshow('My Image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f'Blurred background of {image_file_name}:')
        




OVERLAY_COLOR = (100, 100, 0) # cyan

# Create the segmenter
with python.vision.InteractiveSegmenter.create_from_options(options) as segmenter:

  # Loop through available image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe Image
    image = mp.Image.create_from_file(image_file_name)

    # Retrieve the category masks for the image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x, y))
    segmentation_result = segmenter.segment(image, roi)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image to RGB
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

    # Create an overlay image with the desired color (e.g., (255, 0, 0) for red)
    overlay_image = np.zeros(image_data.shape, dtype=np.uint8)
    overlay_image[:] = OVERLAY_COLOR

    # Create the condition from the category_masks array
    alpha = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1

    # Create an alpha channel from the condition with the desired opacity (e.g., 0.7 for 70%)
    alpha = alpha.astype(float) * 0.7

    # Blend the original image and the overlay image based on the alpha channel
    output_image = image_data * (1 - alpha) + overlay_image * alpha
    output_image = output_image.astype(np.uint8)

    # Draw a white dot with black border to denote the point of interest
    thickness, radius = 6, -1
    keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
    cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
    cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)
    cv2.imshow('My Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'{image_file_name}:')
    