import timm

print(timm.list_models('*bit*', pretrained=True))
model = timm.create_model('resnetv2_50x1_bitm_in21k', pretrained=True, num_classes=10)
# print(model)
print(model.default_cfg)
print(model.get_classifier())
