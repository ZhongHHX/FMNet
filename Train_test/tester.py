####
import torch.nn as nn
import torch
class _DCNv2Pooling(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        rois,
        offset,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
    ):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        output, output_count = _backend.dcn_v2_psroi_pooling_forward(
            input,
            rois,
            offset,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _backend.dcn_v2_psroi_pooling_backward(
            grad_output,
            input,
            rois,
            offset,
            output_count,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )

        return grad_input, None, grad_offset, None, None, None, None, None, None, None, None


dcn_v2_pooling = _DCNv2Pooling.apply
