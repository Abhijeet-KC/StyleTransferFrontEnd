import torch
import torch.nn as nn
from .Encoder import DropPath, Mlp, Attention, Attention_Cross

class TransModule_Config():
  def __init__(
    self,
    nlayer=3,
    d_model=768,
    nhead=8,
    mlp_ratio=4,
    qkv_bias=False,
    attn_drop=0.,
    drop=0.,
    drop_path=0.,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
    norm_first=False
  ):
    self.nlayer = nlayer
    self.d_model = d_model
    self.nhead = nhead
    self.mlp_ratio = mlp_ratio
    self.qkv_bias = qkv_bias
    self.attn_drop = attn_drop
    self.drop = drop
    self.drop_path = drop_path
    self.act_layer = act_layer
    self.norm_layer = norm_layer
    self.norm_first = norm_first

class TransformerEncoderLayer(nn.Module):
  """Implemented as vit block in timm
  """
  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0., 
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)

    self.attn = Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)    
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

  def forward(self, x):
    if self.norm_first == True:
      x = x + self.drop_path(self.attn(self.norm1(x)))
      x = x + self.drop_path(self.mlp(self.norm2(x)))
    else:
      x = self.norm1(x + self.drop_path(self.attn(x)))
      x = self.norm2(x + self.drop_path(self.mlp(x)))
    return x


class TransformerDecoderLayer(nn.Module):
  """Transformer Decoder Layer
  """
  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0., 
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)
    
    self.attn1 = Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.attn2 = Attention_Cross(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.norm3 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

  def forward(self, x, y):
    """
      Args:
        x: output of the former layer
        y: memery of the encoder layer
    """
    if self.norm_first == True:
      x = x + self.drop_path(self.attn1(self.norm1(x)))
      x = x + self.drop_path(self.attn2(self.norm2(x), y))
      x = x + self.drop_path(self.mlp(self.norm3(x)))
    else:
      x = self.norm1(x + self.drop_path(self.attn1(x)))
      x = self.norm2(x + self.drop_path(self.attn2(x, y)))
      x = self.norm3(x + self.drop_path(self.mlp(x)))
    return x



class TransModule(nn.Module):
  """The Transfer Module of Style Transfer via Transformer

  Taking Transformer Decoder as the transfer module.

  Args:
    config: The configuration of the transfer module
  """
  def __init__(self, config: TransModule_Config=None):
    super(TransModule, self).__init__()
    self.layers = nn.ModuleList([
      TransformerDecoderLayer(
          d_model=config.d_model,
          nhead=config.nhead,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          attn_drop=config.attn_drop,
          drop=config.drop,
          drop_path=config.drop_path,
          act_layer=config.act_layer,
          norm_layer=config.norm_layer,
          norm_first=config.norm_first
          ) \
      for i in range(config.nlayer)
    ])

  def forward(self, content_feature, style_feature):
    """
    Args:
      content_feature: Content features，for producing Q sequences. Similar to tgt sequences in pytorch. (Tensor,[Batch,sequence,dim])
      style_feature : Style features，for producing K,V sequences.Similar to memory sequences in pytorch.(Tensor,[Batch,sequence,dim])

    Returns:
      Tensor with shape (Batch,sequence,dim)
    """
    for layer in self.layers:
      content_feature = layer(content_feature, style_feature)
    
    return content_feature
    
# Example usage

if __name__ == "__main__":
    config = TransModule_Config(d_model=768, nhead=8, nlayer=3)
    trans_module = TransModule(config)

    # Dummy input tensors
    content_feature = torch.randn(1, 1024, 768)  
    style_feature = torch.randn(1, 1024, 768)   

    # Forward pass
    output = trans_module(content_feature, style_feature)
    print(output.shape) 