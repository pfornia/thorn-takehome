# Senior Machine Learning Engineer - Thorn Homework Assignment

To complete this assignment, you will need a computer with at least 16 GB of memory, a modern quad-core processor, and 10 GB of available storage. A dedicated GPU is not required, although it may reduce how long it takes to complete the assignments. Most Apple Silicon MacBooks made after 2020 meet these requirements.

You may use AI tools to assist with any part of the assignment, except for the write-up: we want to hear your reasoning in your own words. In your write-up, clearly disclose which tools you used, what tasks they assisted with, and how you verified or modified their output. Your submission should reflect your own understanding of the investigation and conclusions.

We have an image-classification model that assigns images to one of three categories: cat, dog, or other.

Users have reported that the model sometimes incorrectly classifies images as cats or dogs. They can’t share their images with us, so we have only their descriptions of the problem. They have described the false positive images as:

- **User Feedback 1:** “I’m seeing a lot of bad predictions where the image is sort of like a grid or collage and they seem to have lots of lines running through them”
- **User Feedback 2:** “A bunch of the images in my review queue have watermarks and text in them and almost none of them are actually actionable.”
- **User Feedback 3:** “The model is flagging lots of messy looking images that are hard to make out and sort of low quality. The predictions are basically never right on these.”

Your task is to use this feedback to reproduce the user reported false-positive behavior by transforming the images in the `images` directory so that they produce false positive predictions for the `dog` or `cat` classes. You should have received:

- A set of benign base images that the model correctly classifies as “other” in the `images` directory
- The image classifier weights (`model.pt`)
- A `model.py` module containing the code required to run the model
- A `tests` directory containing tests to evaluate your modified images

Generate images using transformations that you believe are consistent with the users’ descriptions. Get creative, extra points are awarded for automated pipelines that apply transformations and get incorrect predictions from the model.

Your goal is to produce two to three modified images that definitely do not depict a cat or dog but that the model classifies as cat or dog. The resulting images should be plausible examples of the issue described by users. Do not use adversarial perturbations or modify the model itself, we’ll be checking that your images produce the expected scores.

Your submission should include:

- Two to three images that produce a false-positive dog prediction based on each user’s feedback.
- The original base image associated with each modified image.
- The model’s predicted class and confidence scores before and after modification.
- The code or scripts used to create and evaluate the modified images.
- A written explanation of:
  - How you interpreted the user feedback
  - Which transformations you tested
  - What characteristics appear to trigger the false positives
  - How you would mitigate the issue if found in production
  - How you would design a system to address future false positives
