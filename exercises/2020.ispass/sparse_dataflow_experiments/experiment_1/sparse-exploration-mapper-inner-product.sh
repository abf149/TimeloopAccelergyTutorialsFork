#timeloop-mapper arch/matraptor_like-int16.yaml \
#                arch/components/*.yaml \
#                prob/prob.yaml \
#                mapper/mapper.yaml \
#                constraints/*.yaml


timeloop-mapper arch/sparse-exporation-architecture.yaml arch/components/*.yaml prob/prob.yaml mapper/mapper.yaml constraints/inner-product-arch-constraints.yaml sparse-opt/naive-row-wise-product.yaml  -o output/

#arch/components/*.yaml \
#                map/*
