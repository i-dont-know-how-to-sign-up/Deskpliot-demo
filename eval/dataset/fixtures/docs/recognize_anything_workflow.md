# Recognize Anything Model workflow

The Recognize Anything Model (RAM) is an image-tagging model. Its architecture contains three key modules: an image encoder for feature extraction, an image-tag recognition decoder for tagging, and a text generation encoder-decoder for captioning.

Image features interact with tags through cross-attention layers in the image-tag interaction encoder and recognition decoder. During training, the recognition head predicts tags parsed from text. During inference, the recognition head predicts image tags and uses them as explicit semantic guidance for image captioning.

For open-vocabulary recognition, RAM uses an off-the-shelf CLIP text encoder to encode tags into semantically rich textual label queries. The implementation uses a Swin Transformer image encoder, a two-layer tag recognition decoder, and CLIP image-feature distillation. The paper does not describe this component as BERT or Q-Former.

Source: Recognize Anything: A Strong Image Tagging Model, Section 2.1-2.2, pages 3-4.
