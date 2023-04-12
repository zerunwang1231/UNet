from io import BytesIO

import imageio
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Logger(object):

    def __init__(self, log_dir):
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
         summary = tf.compat.v1.Summary(value=
                                        [tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
         self.writer.add_summary(summary, step)
         self.writer.flush()

    def image_summary(self, tag, image, step):
        s = BytesIO()
        imageio.toimage(image).save(s, format="png")

        # Create an Image object
        img_sum = tf.compat.v1.Summary.Image(
            encoded_image_string=s.getvalue(),
            height=image.shape[0],
            width=image.shape[1],
        )

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, image=img_sum)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
            imageio.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )

            # Create a Summary value
            img_summaries.append(
                tf.compat.v1.Summary.Value(tag="{}/{}".format(tag, i), image=img_sum)
            )

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
