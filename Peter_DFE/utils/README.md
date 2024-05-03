# DFE Utilities

## Summary

This folder contains two utility files, utils_Peter.py and utils_joscha.py, which provide various helper functions for the DFE torch and DFE tensorflow.

### utils_Peter.py

- sorted_alphanumeric(data): Sorts filenames in alphanumeric order.

- convertRGB2CIELab(image): Converts uint8 RGB image to float32 CIELAB format.

- gauss_noise_tensor(img): Adds Gaussian noise to a torch tensor image.

- get_pts_local_window(center, neighborhood): Generates a local window of points around a center point.

- find_min_surface(A, B): Finds the minimum surface by solving the least-squares problem AxC=B.

- quadratic_interpolation_deep_encodings(center, window_pts, ssr): Performs quadratic interpolation on deep encodings.

- predict(model, data_loader, device): Performs predictions using a trained model on a data loader.

### utils_joscha.py

- sorted_alphanumeric(data): Sorts filenames in alphanumeric order.

- create_relabeling_img(idx, path, filenames, new_size, org_label, window_size): Concatenates reference image with another image from the dataset.

- create_ref_crop_mosaic(img, window_size, bbox_center, gap): Creates a mosaic of the original image and zoomed-in crop.

- get_sift_descriptor(img, px, py): Computes the SIFT descriptor for a given point.

- get_surf_descriptor(img, px, py): Computes the SURF descriptor for a given point.

- get_image_crops_locally(img, window_size, stride, pred): Extracts image crops locally around a predicted point.

- get_image_crops(img, window_size, stride): Extracts all image crops from an image.

- preprocess_deep_encoding_img(path, filename, new_size): Preprocesses an image for deep encoding.

- convertRGB2CIELab(image): Converts uint8 RGB image to float32 CIELAB format.

- get_deep_encodings_descriptor(img, px, py, encoder, window_size): Computes the deep encoding descriptor for a given point.

- std_norm(x, mu, sigma): Performs standard normalization.

- min_max_norm(x, xmax, xmin): Performs min-max normalization.

- get_pts_local_window(center, neighborhood): Generates a local window of points around a center point.

- quadratic_interpolation_deep_encodings(center, img, neighborhood, model, ref_encoding): Performs quadratic interpolation on deep encodings.

- quadratic_interpolation_sift_surf(center, img, neighborhood, method, ref_descriptor): Performs quadratic interpolation using SIFT or SURF descriptors.

- quadratic_interpolation_tm(center, img, neighborhood, ref_crop): Performs quadratic interpolation using template matching.

- find_min_surface(A, B): Finds the minimum surface by solving the least-squares problem AxC=B.

- get_gradients_ssr(A, c, ssr, mode): Computes gradients with respect to SSR values.

- draw_salient_features_dfe(img, window_size, pos): Draws salient features on an image.

