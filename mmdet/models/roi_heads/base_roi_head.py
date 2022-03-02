from abc import ABCMeta, abstractmethod

import torch.nn as nn

from ..builder import build_shared_head


class BaseRoIHead(nn.Module, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_offset(self):
        return hasattr(self, 'offset_head') and self.offset_head is not None

    @property
    def with_footprint(self):
        return hasattr(self, 'footprint_head') and self.footprint_head is not None

    @property
    def with_angle(self):
        return hasattr(self, 'angle_head') and self.angle_head is not None

    @property
    def with_height(self):
        return hasattr(self, 'height_head') and self.height_head is not None

    @property
    def with_offset_height(self):
        return hasattr(self, 'offset_height_head') and self.offset_height_head is not None
    
    @property
    def with_edge(self):
        return hasattr(self, 'edge_head') and self.edge_head is not None

    @property
    def with_side_face(self):
        return hasattr(self, 'side_face_head') and self.side_face_head is not None

    @property
    def with_offset_field(self):
        return hasattr(self, 'offset_field_head') and self.offset_field_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    # @abstractmethod
    # def init_offset_head(self):
    #     pass

    # @abstractmethod
    # def init_angle_head(self):
    #     pass

    # @abstractmethod
    # def init_height_head(self):
    #     pass

    # @abstractmethod
    # def init_offset_height_head(self):
    #     pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""
        pass

    async def async_simple_test(self, x, img_meta, **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    abstractmethod,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        pass

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass
