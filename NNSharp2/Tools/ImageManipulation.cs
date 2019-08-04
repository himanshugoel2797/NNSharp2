using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.Tools
{
    public class ImageManipulation
    {
        public static Bitmap ScaleImage(NRandom rsz_rng, Image image, int width, int height)
        {
            float aspectRatio = (float)image.Width / image.Height;

            float height_var = (height - (int)(height * aspectRatio)) * 0.8f;
            float width_var = (width - (int)(width / aspectRatio)) * 0.8f;

            var destRect = new Rectangle((int)(height_var * rsz_rng.NextDouble()), 0, (int)(height * aspectRatio), width);
            if (aspectRatio > 1.0f)
                destRect = new Rectangle(0, (int)(width_var * rsz_rng.NextDouble()), height, (int)(width / aspectRatio));
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static string[] CutImage(Image image, int width, int height, string dst, string filename)
        {
            List<string> cut_paths = new List<string>();

            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            //destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            for (int y = 0; y < image.Height - height; y += height / 2)
                for (int x = 0; x < image.Width - width; x += width / 2)
                {
                    using (var graphics = Graphics.FromImage(destImage))
                    {
                        graphics.CompositingMode = CompositingMode.SourceCopy;
                        graphics.CompositingQuality = CompositingQuality.HighQuality;
                        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                        graphics.SmoothingMode = SmoothingMode.HighQuality;
                        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                        using (var wrapMode = new ImageAttributes())
                        {
                            wrapMode.SetWrapMode(WrapMode.Clamp);
                            //wrapMode.
                            graphics.DrawImage(image, destRect, x, y, width, height, GraphicsUnit.Pixel, wrapMode);
                        }
                    }

                    string path = Path.Combine(dst, filename + "_" + x + "_" + y + ".png");
                    cut_paths.Add(path);
                    destImage.Save(path);
                }

            destImage.Dispose();
            return cut_paths.ToArray();
        }

        public static void SaveImage(string file, Tensor img_vec, int Side)
        {
            var bmp = new Bitmap(Side, Side);
            float[] img = new float[img_vec.Axes.Aggregate((a, b) => a * b)];
            //img_vec.Read(img);

            float max = float.MinValue;
            float min = float.MaxValue;

            for (int i = 0; i < img.Length; i++)
            {
                if (img[i] > max)
                    max = img[i];

                if (img[i] < min)
                    min = img[i];

                img[i] = 0.5f * img[i] + 0.5f;
            }

            try
            {
                for (int h = 0; h < bmp.Height; h++)
                    for (int w = 0; w < bmp.Width; w++)
                    {
                        //img[h * bmp.Width + w] = (img[h * bmp.Width + w] - min) / (max - min);
                        //img[bmp.Width * bmp.Height + h * bmp.Width + w] = (img[bmp.Width * bmp.Height + h * bmp.Width + w] - min) / (max - min);
                        //img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] = (img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] - min) / (max - min);

                        bmp.SetPixel(w, h, Color.FromArgb((int)(img[h * bmp.Width + w] * 255.0f), (int)(img[bmp.Width * bmp.Height + h * bmp.Width + w] * 255.0f), (int)(img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] * 255.0f)));
                    }
            }
            catch (Exception) { }

            bmp.Save(file);
            bmp.Dispose();
        }
    }
}
