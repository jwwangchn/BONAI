# from bstool import rotate_iou
from .builder import IOU_CALCULATORS


def rotate_iou(bboxes1, bboxes2):
    return None

@IOU_CALCULATORS.register_module()
class RBboxOverlaps2D(object):
    """2D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 5) in <xc, yc, w, h, alpha>
                format, or shape (m, 6) in <xc, yc, w, h, alpha, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 5) in <xc, yc, w, h, alpha>
                format, shape (m, 6) in <xc, yc, w, h, alpha, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return rbbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def rbbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimenstion is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)
    if is_aligned:
        ious = rotate_iou(bboxes1, bboxes2) #(m,1)
        return ious.diagonal()
    else:
        ious = rotate_iou(bboxes1, bboxes2) #(m,n)
    return ious
