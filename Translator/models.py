import os
import shutil

from django.db import models
from django.db.models.signals import pre_delete, pre_save
from django.dispatch import receiver
from django.conf import settings

# # Create your models here.

def get_upload_path(instance, filename):
    return os.path.join("videos", filename)


class Video(models.Model):
    title = models.CharField(max_length=200)
    video_file = models.FileField(upload_to=get_upload_path)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        self.title = self.title.lower()  # Convert title to lowercase

        if self.pk:
            # Get the previous object
            previous_obj = Video.objects.get(pk=self.pk)

            # Check if the video file has changed
            if previous_obj.video_file != self.video_file:
                # Delete the previous video file
                previous_obj.video_file.delete(save=False)

        super().save(*args, **kwargs)

        # Rename the video file based on the title if it exists
        if self.video_file:
            file_path = os.path.join(settings.MEDIA_ROOT, self.video_file.name)

            # Check if the file exists
            if os.path.exists(file_path):
                # Extract the file extension
                file_extension = os.path.splitext(file_path)[1]
                new_filename = self.title + file_extension
                new_path = os.path.join(settings.MEDIA_ROOT, "videos", new_filename)

                # Rename the file
                shutil.move(file_path, new_path)

                # Update the file field with the new name
                self.video_file.name = os.path.join("videos", new_filename)

        super().save(*args, **kwargs)


@receiver(pre_delete, sender=Video)
def delete_video_file(sender, instance, **kwargs):
    instance.video_file.delete(False)


@receiver(pre_save, sender=Video)
def delete_previous_video_file(sender, instance, **kwargs):
    if instance.pk:
        previous_obj = Video.objects.get(pk=instance.pk)
        if previous_obj.video_file != instance.video_file:
            previous_obj.video_file.delete(save=False)
