ше
Ћ§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8ДЕ

le_net5half/conv2d/kernelVarHandleOp*
shape:**
shared_namele_net5half/conv2d/kernel*
dtype0*
_output_shapes
: 

-le_net5half/conv2d/kernel/Read/ReadVariableOpReadVariableOple_net5half/conv2d/kernel*
dtype0*&
_output_shapes
:

le_net5half/conv2d/biasVarHandleOp*
shape:*(
shared_namele_net5half/conv2d/bias*
dtype0*
_output_shapes
: 

+le_net5half/conv2d/bias/Read/ReadVariableOpReadVariableOple_net5half/conv2d/bias*
dtype0*
_output_shapes
:

le_net5half/conv2d_1/kernelVarHandleOp*
shape:*,
shared_namele_net5half/conv2d_1/kernel*
dtype0*
_output_shapes
: 

/le_net5half/conv2d_1/kernel/Read/ReadVariableOpReadVariableOple_net5half/conv2d_1/kernel*
dtype0*&
_output_shapes
:

le_net5half/conv2d_1/biasVarHandleOp*
shape:**
shared_namele_net5half/conv2d_1/bias*
dtype0*
_output_shapes
: 

-le_net5half/conv2d_1/bias/Read/ReadVariableOpReadVariableOple_net5half/conv2d_1/bias*
dtype0*
_output_shapes
:

le_net5half/conv2d_2/kernelVarHandleOp*
shape:<*,
shared_namele_net5half/conv2d_2/kernel*
dtype0*
_output_shapes
: 

/le_net5half/conv2d_2/kernel/Read/ReadVariableOpReadVariableOple_net5half/conv2d_2/kernel*
dtype0*&
_output_shapes
:<

le_net5half/conv2d_2/biasVarHandleOp*
shape:<**
shared_namele_net5half/conv2d_2/bias*
dtype0*
_output_shapes
: 

-le_net5half/conv2d_2/bias/Read/ReadVariableOpReadVariableOple_net5half/conv2d_2/bias*
dtype0*
_output_shapes
:<

le_net5half/dense/kernelVarHandleOp*
shape
:<**)
shared_namele_net5half/dense/kernel*
dtype0*
_output_shapes
: 

,le_net5half/dense/kernel/Read/ReadVariableOpReadVariableOple_net5half/dense/kernel*
dtype0*
_output_shapes

:<*

le_net5half/dense/biasVarHandleOp*
shape:**'
shared_namele_net5half/dense/bias*
dtype0*
_output_shapes
: 
}
*le_net5half/dense/bias/Read/ReadVariableOpReadVariableOple_net5half/dense/bias*
dtype0*
_output_shapes
:*

le_net5half/dense_1/kernelVarHandleOp*
shape
:*
*+
shared_namele_net5half/dense_1/kernel*
dtype0*
_output_shapes
: 

.le_net5half/dense_1/kernel/Read/ReadVariableOpReadVariableOple_net5half/dense_1/kernel*
dtype0*
_output_shapes

:*


le_net5half/dense_1/biasVarHandleOp*
shape:
*)
shared_namele_net5half/dense_1/bias*
dtype0*
_output_shapes
: 

,le_net5half/dense_1/bias/Read/ReadVariableOpReadVariableOple_net5half/dense_1/bias*
dtype0*
_output_shapes
:


NoOpNoOp
Ш
ConstConst"/device:CPU:0*
valueљBі Bя
Г
	conv1
	max_pool1
	conv2
	max_pool2
	conv3
fc1
fc2
trainable_variables
		variables

regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
h

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
F
0
1
2
3
!4
"5
'6
(7
-8
.9
F
0
1
2
3
!4
"5
'6
(7
-8
.9
 

3non_trainable_variables

4layers
5metrics
trainable_variables
		variables

regularization_losses
6layer_regularization_losses
 
VT
VARIABLE_VALUEle_net5half/conv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEle_net5half/conv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1

7non_trainable_variables

8layers
9metrics
	variables
regularization_losses
trainable_variables
:layer_regularization_losses
 
 
 

;non_trainable_variables

<layers
=metrics
	variables
regularization_losses
trainable_variables
>layer_regularization_losses
XV
VARIABLE_VALUEle_net5half/conv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEle_net5half/conv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1

?non_trainable_variables

@layers
Ametrics
	variables
regularization_losses
trainable_variables
Blayer_regularization_losses
 
 
 

Cnon_trainable_variables

Dlayers
Emetrics
	variables
regularization_losses
trainable_variables
Flayer_regularization_losses
XV
VARIABLE_VALUEle_net5half/conv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEle_net5half/conv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1

Gnon_trainable_variables

Hlayers
Imetrics
#	variables
$regularization_losses
%trainable_variables
Jlayer_regularization_losses
SQ
VARIABLE_VALUEle_net5half/dense/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEle_net5half/dense/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1

Knon_trainable_variables

Llayers
Mmetrics
)	variables
*regularization_losses
+trainable_variables
Nlayer_regularization_losses
US
VARIABLE_VALUEle_net5half/dense_1/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEle_net5half/dense_1/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1

Onon_trainable_variables

Players
Qmetrics
/	variables
0regularization_losses
1trainable_variables
Rlayer_regularization_losses
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 

serving_default_input_1Placeholder*$
shape:џџџџџџџџџ  *
dtype0*/
_output_shapes
:џџџџџџџџџ  
Е
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1le_net5half/conv2d/kernelle_net5half/conv2d/biasle_net5half/conv2d_1/kernelle_net5half/conv2d_1/biasle_net5half/conv2d_2/kernelle_net5half/conv2d_2/biasle_net5half/dense/kernelle_net5half/dense/biasle_net5half/dense_1/kernelle_net5half/dense_1/bias*+
_gradient_op_typePartitionedCall-7283*+
f&R$
"__inference_signature_wrapper_7168*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ

O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
р
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-le_net5half/conv2d/kernel/Read/ReadVariableOp+le_net5half/conv2d/bias/Read/ReadVariableOp/le_net5half/conv2d_1/kernel/Read/ReadVariableOp-le_net5half/conv2d_1/bias/Read/ReadVariableOp/le_net5half/conv2d_2/kernel/Read/ReadVariableOp-le_net5half/conv2d_2/bias/Read/ReadVariableOp,le_net5half/dense/kernel/Read/ReadVariableOp*le_net5half/dense/bias/Read/ReadVariableOp.le_net5half/dense_1/kernel/Read/ReadVariableOp,le_net5half/dense_1/bias/Read/ReadVariableOpConst*+
_gradient_op_typePartitionedCall-7315*&
f!R
__inference__traced_save_7314*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamele_net5half/conv2d/kernelle_net5half/conv2d/biasle_net5half/conv2d_1/kernelle_net5half/conv2d_1/biasle_net5half/conv2d_2/kernelle_net5half/conv2d_2/biasle_net5half/dense/kernelle_net5half/dense/biasle_net5half/dense_1/kernelle_net5half/dense_1/bias*+
_gradient_op_typePartitionedCall-7358*)
f$R"
 __inference__traced_restore_7357*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: Жѓ
ш
Ї
&__inference_dense_1_layer_call_fn_7259

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-7117*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7111*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ

IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ*::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
К!
Т
__inference__traced_save_7314
file_prefix8
4savev2_le_net5half_conv2d_kernel_read_readvariableop6
2savev2_le_net5half_conv2d_bias_read_readvariableop:
6savev2_le_net5half_conv2d_1_kernel_read_readvariableop8
4savev2_le_net5half_conv2d_1_bias_read_readvariableop:
6savev2_le_net5half_conv2d_2_kernel_read_readvariableop8
4savev2_le_net5half_conv2d_2_bias_read_readvariableop7
3savev2_le_net5half_dense_kernel_read_readvariableop5
1savev2_le_net5half_dense_bias_read_readvariableop9
5savev2_le_net5half_dense_1_kernel_read_readvariableop7
3savev2_le_net5half_dense_1_bias_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_f1950f655eab4d69ab77d3c367d017ae/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: є
SaveV2/tensor_namesConst"/device:CPU:0*
valueB
B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

SaveV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:
Х
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_le_net5half_conv2d_kernel_read_readvariableop2savev2_le_net5half_conv2d_bias_read_readvariableop6savev2_le_net5half_conv2d_1_kernel_read_readvariableop4savev2_le_net5half_conv2d_1_bias_read_readvariableop6savev2_le_net5half_conv2d_2_kernel_read_readvariableop4savev2_le_net5half_conv2d_2_bias_read_readvariableop3savev2_le_net5half_dense_kernel_read_readvariableop1savev2_le_net5half_dense_bias_read_readvariableop5savev2_le_net5half_dense_1_kernel_read_readvariableop3savev2_le_net5half_dense_1_bias_read_readvariableop"/device:CPU:0*
dtypes
2
*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:У
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapesn
l: :::::<:<:<*:*:*
:
: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : :
 

c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_6920

inputs
identityЂ
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6962

inputs
identityЂ
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
 
о
?__inference_dense_layer_call_and_return_conditional_losses_7204

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpЈ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:<*X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:c
Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџk
Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:<*j
Tensordot/Reshape_1/shapeConst*
valueB"<   *   *
dtype0*
_output_shapes
:
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:<*
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*[
Tensordot/Const_2Const*
valueB:**
dtype0*
_output_shapes
:Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ* 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:*
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ*"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ<::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 

л
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6985

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:<Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ< 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:<
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
э
Э
*__inference_le_net5half_layer_call_fn_7149
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*+
_gradient_op_typePartitionedCall-7136*N
fIRG
E__inference_le_net5half_layer_call_and_return_conditional_losses_7130*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ

IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџ  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: : :	 : :
 

л
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6943

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ш%
р
A__inference_dense_1_layer_call_and_return_conditional_losses_7111

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpЈ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:*
X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:c
Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџk
Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:*
j
Tensordot/Reshape_1/shapeConst*
valueB"*   
   *
dtype0*
_output_shapes
:
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:*

Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
[
Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:

BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
`
Max/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:џџџџџџџџџd
subSubBiasAdd:output:0Max:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
M
ExpExpsub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
`
Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: ~
SumSumExp:y:0Sum/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:џџџџџџџџџc
truedivRealDivExp:y:0Sum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

IdentityIdentitytruediv:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ*::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 

І
%__inference_conv2d_layer_call_fn_6912

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6907*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_6901*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 

й
@__inference_conv2d_layer_call_and_return_conditional_losses_6901

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ф
Ѕ
$__inference_dense_layer_call_fn_7211

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-7058*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7052*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ*
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ*"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ<::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
 
о
?__inference_dense_layer_call_and_return_conditional_losses_7052

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpЈ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:<*X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:c
Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџk
Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:<*j
Tensordot/Reshape_1/shapeConst*
valueB"<   *   *
dtype0*
_output_shapes
:
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:<*
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*[
Tensordot/Const_2Const*
valueB:**
dtype0*
_output_shapes
:Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ* 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:*
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ*"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ<::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ш%
р
A__inference_dense_1_layer_call_and_return_conditional_losses_7252

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpЈ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:*
X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:c
Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџk
Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:*
j
Tensordot/Reshape_1/shapeConst*
valueB"*   
   *
dtype0*
_output_shapes
:
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:*

Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
[
Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:

BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
`
Max/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:џџџџџџџџџd
subSubBiasAdd:output:0Max:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
M
ExpExpsub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
`
Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: ~
SumSumExp:y:0Sum/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:џџџџџџџџџc
truedivRealDivExp:y:0Sum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

IdentityIdentitytruediv:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ*::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
,
Ё
 __inference__traced_restore_7357
file_prefix.
*assignvariableop_le_net5half_conv2d_kernel.
*assignvariableop_1_le_net5half_conv2d_bias2
.assignvariableop_2_le_net5half_conv2d_1_kernel0
,assignvariableop_3_le_net5half_conv2d_1_bias2
.assignvariableop_4_le_net5half_conv2d_2_kernel0
,assignvariableop_5_le_net5half_conv2d_2_bias/
+assignvariableop_6_le_net5half_dense_kernel-
)assignvariableop_7_le_net5half_dense_bias1
-assignvariableop_8_le_net5half_dense_1_kernel/
+assignvariableop_9_le_net5half_dense_1_bias
identity_11ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1ї
RestoreV2/tensor_namesConst"/device:CPU:0*
valueB
B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

RestoreV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:
а
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2
*<
_output_shapes*
(::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp*assignvariableop_le_net5half_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp*assignvariableop_1_le_net5half_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp.assignvariableop_2_le_net5half_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp,assignvariableop_3_le_net5half_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp.assignvariableop_4_le_net5half_conv2d_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp,assignvariableop_5_le_net5half_conv2d_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp+assignvariableop_6_le_net5half_dense_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp)assignvariableop_7_le_net5half_dense_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp-assignvariableop_8_le_net5half_dense_1_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp+assignvariableop_9_le_net5half_dense_1_biasIdentity_9:output:0*
dtype0*
_output_shapes
 
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ћ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: И
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6: : : : : :+ '
%
_user_specified_namefile_prefix: : :	 : :
 
Ї
J
.__inference_max_pooling2d_1_layer_call_fn_6971

inputs
identityР
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-6968*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6962*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
Ѓ
H
,__inference_max_pooling2d_layer_call_fn_6929

inputs
identityО
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-6926*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_6920*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
ь)
Ф
E__inference_le_net5half_layer_call_and_return_conditional_losses_7130
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6907*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_6901*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ
conv2d/IdentityIdentity'conv2d/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџУ
max_pooling2d/PartitionedCallPartitionedCallconv2d/Identity:output:0*+
_gradient_op_typePartitionedCall-6926*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_6920*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ
max_pooling2d/IdentityIdentity&max_pooling2d/PartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallmax_pooling2d/Identity:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6949*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6943*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ

Ѕ
conv2d_1/IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ

Щ
max_pooling2d_1/PartitionedCallPartitionedCallconv2d_1/Identity:output:0*+
_gradient_op_typePartitionedCall-6968*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6962*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ
max_pooling2d_1/IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџІ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!max_pooling2d_1/Identity:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6991*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6985*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ<Ѕ
conv2d_2/IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ<
dense/StatefulPartitionedCallStatefulPartitionedCallconv2d_2/Identity:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-7058*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7052*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ*
dense/IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ*
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense/Identity:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-7117*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7111*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ
Ђ
dense_1/IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ

IdentityIdentitydense_1/Identity:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџ  ::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: : :	 : :
 
 
Ј
'__inference_conv2d_1_layer_call_fn_6954

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6949*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6943*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
П
Х
"__inference_signature_wrapper_7168
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*+
_gradient_op_typePartitionedCall-7155*(
f#R!
__inference__wrapped_model_6887*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ

IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџ  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: : :	 : :
 

Й
__inference__wrapped_model_6887
input_15
1le_net5half_conv2d_conv2d_readvariableop_resource6
2le_net5half_conv2d_biasadd_readvariableop_resource7
3le_net5half_conv2d_1_conv2d_readvariableop_resource8
4le_net5half_conv2d_1_biasadd_readvariableop_resource7
3le_net5half_conv2d_2_conv2d_readvariableop_resource8
4le_net5half_conv2d_2_biasadd_readvariableop_resource7
3le_net5half_dense_tensordot_readvariableop_resource5
1le_net5half_dense_biasadd_readvariableop_resource9
5le_net5half_dense_1_tensordot_readvariableop_resource7
3le_net5half_dense_1_biasadd_readvariableop_resource
identityЂ)le_net5half/conv2d/BiasAdd/ReadVariableOpЂ(le_net5half/conv2d/Conv2D/ReadVariableOpЂ+le_net5half/conv2d_1/BiasAdd/ReadVariableOpЂ*le_net5half/conv2d_1/Conv2D/ReadVariableOpЂ+le_net5half/conv2d_2/BiasAdd/ReadVariableOpЂ*le_net5half/conv2d_2/Conv2D/ReadVariableOpЂ(le_net5half/dense/BiasAdd/ReadVariableOpЂ*le_net5half/dense/Tensordot/ReadVariableOpЂ*le_net5half/dense_1/BiasAdd/ReadVariableOpЂ,le_net5half/dense_1/Tensordot/ReadVariableOpа
(le_net5half/conv2d/Conv2D/ReadVariableOpReadVariableOp1le_net5half_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:С
le_net5half/conv2d/Conv2DConv2Dinput_10le_net5half/conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:џџџџџџџџџЦ
)le_net5half/conv2d/BiasAdd/ReadVariableOpReadVariableOp2le_net5half_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ж
le_net5half/conv2d/BiasAddBiasAdd"le_net5half/conv2d/Conv2D:output:01le_net5half/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ~
le_net5half/conv2d/ReluRelu#le_net5half/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџР
!le_net5half/max_pooling2d/MaxPoolMaxPool%le_net5half/conv2d/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџд
*le_net5half/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3le_net5half_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:ш
le_net5half/conv2d_1/Conv2DConv2D*le_net5half/max_pooling2d/MaxPool:output:02le_net5half/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ

Ъ
+le_net5half/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4le_net5half_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:М
le_net5half/conv2d_1/BiasAddBiasAdd$le_net5half/conv2d_1/Conv2D:output:03le_net5half/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ


le_net5half/conv2d_1/ReluRelu%le_net5half/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

Ф
#le_net5half/max_pooling2d_1/MaxPoolMaxPool'le_net5half/conv2d_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџд
*le_net5half/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3le_net5half_conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:<ъ
le_net5half/conv2d_2/Conv2DConv2D,le_net5half/max_pooling2d_1/MaxPool:output:02le_net5half/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ<Ъ
+le_net5half/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4le_net5half_conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:<М
le_net5half/conv2d_2/BiasAddBiasAdd$le_net5half/conv2d_2/Conv2D:output:03le_net5half/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<
le_net5half/conv2d_2/ReluRelu%le_net5half/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<Ь
*le_net5half/dense/Tensordot/ReadVariableOpReadVariableOp3le_net5half_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:<*j
 le_net5half/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:u
 le_net5half/dense/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:x
!le_net5half/dense/Tensordot/ShapeShape'le_net5half/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:k
)le_net5half/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
$le_net5half/dense/Tensordot/GatherV2GatherV2*le_net5half/dense/Tensordot/Shape:output:0)le_net5half/dense/Tensordot/free:output:02le_net5half/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+le_net5half/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
&le_net5half/dense/Tensordot/GatherV2_1GatherV2*le_net5half/dense/Tensordot/Shape:output:0)le_net5half/dense/Tensordot/axes:output:04le_net5half/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!le_net5half/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:Є
 le_net5half/dense/Tensordot/ProdProd-le_net5half/dense/Tensordot/GatherV2:output:0*le_net5half/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#le_net5half/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:Њ
"le_net5half/dense/Tensordot/Prod_1Prod/le_net5half/dense/Tensordot/GatherV2_1:output:0,le_net5half/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'le_net5half/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ф
"le_net5half/dense/Tensordot/concatConcatV2)le_net5half/dense/Tensordot/free:output:0)le_net5half/dense/Tensordot/axes:output:00le_net5half/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:Џ
!le_net5half/dense/Tensordot/stackPack)le_net5half/dense/Tensordot/Prod:output:0+le_net5half/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:Т
%le_net5half/dense/Tensordot/transpose	Transpose'le_net5half/conv2d_2/Relu:activations:0+le_net5half/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<Р
#le_net5half/dense/Tensordot/ReshapeReshape)le_net5half/dense/Tensordot/transpose:y:0*le_net5half/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ}
,le_net5half/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:Ш
'le_net5half/dense/Tensordot/transpose_1	Transpose2le_net5half/dense/Tensordot/ReadVariableOp:value:05le_net5half/dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:<*|
+le_net5half/dense/Tensordot/Reshape_1/shapeConst*
valueB"<   *   *
dtype0*
_output_shapes
:М
%le_net5half/dense/Tensordot/Reshape_1Reshape+le_net5half/dense/Tensordot/transpose_1:y:04le_net5half/dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:<*М
"le_net5half/dense/Tensordot/MatMulMatMul,le_net5half/dense/Tensordot/Reshape:output:0.le_net5half/dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*m
#le_net5half/dense/Tensordot/Const_2Const*
valueB:**
dtype0*
_output_shapes
:k
)le_net5half/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: я
$le_net5half/dense/Tensordot/concat_1ConcatV2-le_net5half/dense/Tensordot/GatherV2:output:0,le_net5half/dense/Tensordot/Const_2:output:02le_net5half/dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:Н
le_net5half/dense/TensordotReshape,le_net5half/dense/Tensordot/MatMul:product:0-le_net5half/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*Ф
(le_net5half/dense/BiasAdd/ReadVariableOpReadVariableOp1le_net5half_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:*Ж
le_net5half/dense/BiasAddBiasAdd$le_net5half/dense/Tensordot:output:00le_net5half/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*а
,le_net5half/dense_1/Tensordot/ReadVariableOpReadVariableOp5le_net5half_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:*
l
"le_net5half/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:w
"le_net5half/dense_1/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:u
#le_net5half/dense_1/Tensordot/ShapeShape"le_net5half/dense/BiasAdd:output:0*
T0*
_output_shapes
:m
+le_net5half/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
&le_net5half/dense_1/Tensordot/GatherV2GatherV2,le_net5half/dense_1/Tensordot/Shape:output:0+le_net5half/dense_1/Tensordot/free:output:04le_net5half/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-le_net5half/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
(le_net5half/dense_1/Tensordot/GatherV2_1GatherV2,le_net5half/dense_1/Tensordot/Shape:output:0+le_net5half/dense_1/Tensordot/axes:output:06le_net5half/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#le_net5half/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:Њ
"le_net5half/dense_1/Tensordot/ProdProd/le_net5half/dense_1/Tensordot/GatherV2:output:0,le_net5half/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%le_net5half/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:А
$le_net5half/dense_1/Tensordot/Prod_1Prod1le_net5half/dense_1/Tensordot/GatherV2_1:output:0.le_net5half/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)le_net5half/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ь
$le_net5half/dense_1/Tensordot/concatConcatV2+le_net5half/dense_1/Tensordot/free:output:0+le_net5half/dense_1/Tensordot/axes:output:02le_net5half/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:Е
#le_net5half/dense_1/Tensordot/stackPack+le_net5half/dense_1/Tensordot/Prod:output:0-le_net5half/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:С
'le_net5half/dense_1/Tensordot/transpose	Transpose"le_net5half/dense/BiasAdd:output:0-le_net5half/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*Ц
%le_net5half/dense_1/Tensordot/ReshapeReshape+le_net5half/dense_1/Tensordot/transpose:y:0,le_net5half/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
.le_net5half/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:Ю
)le_net5half/dense_1/Tensordot/transpose_1	Transpose4le_net5half/dense_1/Tensordot/ReadVariableOp:value:07le_net5half/dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:*
~
-le_net5half/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"*   
   *
dtype0*
_output_shapes
:Т
'le_net5half/dense_1/Tensordot/Reshape_1Reshape-le_net5half/dense_1/Tensordot/transpose_1:y:06le_net5half/dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:*
Т
$le_net5half/dense_1/Tensordot/MatMulMatMul.le_net5half/dense_1/Tensordot/Reshape:output:00le_net5half/dense_1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
o
%le_net5half/dense_1/Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:m
+le_net5half/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ї
&le_net5half/dense_1/Tensordot/concat_1ConcatV2/le_net5half/dense_1/Tensordot/GatherV2:output:0.le_net5half/dense_1/Tensordot/Const_2:output:04le_net5half/dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:У
le_net5half/dense_1/TensordotReshape.le_net5half/dense_1/Tensordot/MatMul:product:0/le_net5half/dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
Ш
*le_net5half/dense_1/BiasAdd/ReadVariableOpReadVariableOp3le_net5half_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
М
le_net5half/dense_1/BiasAddBiasAdd&le_net5half/dense_1/Tensordot:output:02le_net5half/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
t
)le_net5half/dense_1/Max/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: У
le_net5half/dense_1/MaxMax$le_net5half/dense_1/BiasAdd:output:02le_net5half/dense_1/Max/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:џџџџџџџџџ 
le_net5half/dense_1/subSub$le_net5half/dense_1/BiasAdd:output:0 le_net5half/dense_1/Max:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
u
le_net5half/dense_1/ExpExple_net5half/dense_1/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
t
)le_net5half/dense_1/Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: К
le_net5half/dense_1/SumSumle_net5half/dense_1/Exp:y:02le_net5half/dense_1/Sum/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:џџџџџџџџџ
le_net5half/dense_1/truedivRealDivle_net5half/dense_1/Exp:y:0 le_net5half/dense_1/Sum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
А
IdentityIdentityle_net5half/dense_1/truediv:z:0*^le_net5half/conv2d/BiasAdd/ReadVariableOp)^le_net5half/conv2d/Conv2D/ReadVariableOp,^le_net5half/conv2d_1/BiasAdd/ReadVariableOp+^le_net5half/conv2d_1/Conv2D/ReadVariableOp,^le_net5half/conv2d_2/BiasAdd/ReadVariableOp+^le_net5half/conv2d_2/Conv2D/ReadVariableOp)^le_net5half/dense/BiasAdd/ReadVariableOp+^le_net5half/dense/Tensordot/ReadVariableOp+^le_net5half/dense_1/BiasAdd/ReadVariableOp-^le_net5half/dense_1/Tensordot/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџ  ::::::::::2X
*le_net5half/conv2d_1/Conv2D/ReadVariableOp*le_net5half/conv2d_1/Conv2D/ReadVariableOp2X
*le_net5half/dense/Tensordot/ReadVariableOp*le_net5half/dense/Tensordot/ReadVariableOp2Z
+le_net5half/conv2d_2/BiasAdd/ReadVariableOp+le_net5half/conv2d_2/BiasAdd/ReadVariableOp2T
(le_net5half/dense/BiasAdd/ReadVariableOp(le_net5half/dense/BiasAdd/ReadVariableOp2T
(le_net5half/conv2d/Conv2D/ReadVariableOp(le_net5half/conv2d/Conv2D/ReadVariableOp2Z
+le_net5half/conv2d_1/BiasAdd/ReadVariableOp+le_net5half/conv2d_1/BiasAdd/ReadVariableOp2\
,le_net5half/dense_1/Tensordot/ReadVariableOp,le_net5half/dense_1/Tensordot/ReadVariableOp2X
*le_net5half/dense_1/BiasAdd/ReadVariableOp*le_net5half/dense_1/BiasAdd/ReadVariableOp2V
)le_net5half/conv2d/BiasAdd/ReadVariableOp)le_net5half/conv2d/BiasAdd/ReadVariableOp2X
*le_net5half/conv2d_2/Conv2D/ReadVariableOp*le_net5half/conv2d_2/Conv2D/ReadVariableOp: : : : : :' #
!
_user_specified_name	input_1: : :	 : :
 
 
Ј
'__inference_conv2d_2_layer_call_fn_6996

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6991*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6985*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*Л
serving_defaultЇ
C
input_18
serving_default_input_1:0џџџџџџџџџ  D
output_18
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:

	conv1
	max_pool1
	conv2
	max_pool2
	conv3
fc1
fc2
trainable_variables
		variables

regularization_losses
	keras_api

signatures
S__call__
*T&call_and_return_all_conditional_losses
U_default_save_signature"
_tf_keras_modelј{"class_name": "LeNet5Half", "name": "le_net5half", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "LeNet5Half"}}
ч

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"Т
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
љ
	variables
regularization_losses
trainable_variables
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"ъ
_tf_keras_layerа{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ы

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"Ц
_tf_keras_layerЌ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
§
	variables
regularization_losses
trainable_variables
 	keras_api
\__call__
*]&call_and_return_all_conditional_losses"ю
_tf_keras_layerд{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ь

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
^__call__
*_&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 60, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}}
я

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
`__call__
*a&call_and_return_all_conditional_losses"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 42, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}}
є

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
b__call__
*c&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 42}}}}
f
0
1
2
3
!4
"5
'6
(7
-8
.9"
trackable_list_wrapper
f
0
1
2
3
!4
"5
'6
(7
-8
.9"
trackable_list_wrapper
 "
trackable_list_wrapper
З
3non_trainable_variables

4layers
5metrics
trainable_variables
		variables

regularization_losses
6layer_regularization_losses
S__call__
U_default_save_signature
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
,
dserving_default"
signature_map
3:12le_net5half/conv2d/kernel
%:#2le_net5half/conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

7non_trainable_variables

8layers
9metrics
	variables
regularization_losses
trainable_variables
:layer_regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

;non_trainable_variables

<layers
=metrics
	variables
regularization_losses
trainable_variables
>layer_regularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
5:32le_net5half/conv2d_1/kernel
':%2le_net5half/conv2d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

?non_trainable_variables

@layers
Ametrics
	variables
regularization_losses
trainable_variables
Blayer_regularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

Cnon_trainable_variables

Dlayers
Emetrics
	variables
regularization_losses
trainable_variables
Flayer_regularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
5:3<2le_net5half/conv2d_2/kernel
':%<2le_net5half/conv2d_2/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper

Gnon_trainable_variables

Hlayers
Imetrics
#	variables
$regularization_losses
%trainable_variables
Jlayer_regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
*:(<*2le_net5half/dense/kernel
$:"*2le_net5half/dense/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper

Knon_trainable_variables

Llayers
Mmetrics
)	variables
*regularization_losses
+trainable_variables
Nlayer_regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
,:**
2le_net5half/dense_1/kernel
&:$
2le_net5half/dense_1/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper

Onon_trainable_variables

Players
Qmetrics
/	variables
0regularization_losses
1trainable_variables
Rlayer_regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2§
*__inference_le_net5half_layer_call_fn_7149Ю
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_1џџџџџџџџџ  
2
E__inference_le_net5half_layer_call_and_return_conditional_losses_7130Ю
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_1џџџџџџџџџ  
х2т
__inference__wrapped_model_6887О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_1џџџџџџџџџ  
2
%__inference_conv2d_layer_call_fn_6912з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
@__inference_conv2d_layer_call_and_return_conditional_losses_6901з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_max_pooling2d_layer_call_fn_6929р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Џ2Ќ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_6920р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
'__inference_conv2d_1_layer_call_fn_6954з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ё2
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6943з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
.__inference_max_pooling2d_1_layer_call_fn_6971р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Б2Ў
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6962р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
'__inference_conv2d_2_layer_call_fn_6996з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ё2
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6985з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ю2Ы
$__inference_dense_layer_call_fn_7211Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_7204Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_dense_1_layer_call_fn_7259Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_dense_1_layer_call_and_return_conditional_losses_7252Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
1B/
"__inference_signature_wrapper_7168input_1Т
,__inference_max_pooling2d_layer_call_fn_6929RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџе
@__inference_conv2d_layer_call_and_return_conditional_losses_6901IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 з
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6985!"IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
 з
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6943IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
*__inference_le_net5half_layer_call_fn_7149h
!"'(-.8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ  
Њ " џџџџџџџџџ
Џ
?__inference_dense_layer_call_and_return_conditional_losses_7204l'(7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ<
Њ "-Ђ*
# 
0џџџџџџџџџ*
 ъ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_6920RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ­
%__inference_conv2d_layer_call_fn_6912IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЏ
'__inference_conv2d_2_layer_call_fn_6996!"IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<Ї
__inference__wrapped_model_6887
!"'(-.8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ  
Њ ";Њ8
6
output_1*'
output_1џџџџџџџџџ
ь
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6962RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
A__inference_dense_1_layer_call_and_return_conditional_losses_7252l-.7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ*
Њ "-Ђ*
# 
0џџџџџџџџџ

 Ф
.__inference_max_pooling2d_1_layer_call_fn_6971RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџО
E__inference_le_net5half_layer_call_and_return_conditional_losses_7130u
!"'(-.8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ  
Њ "-Ђ*
# 
0џџџџџџџџџ

 
$__inference_dense_layer_call_fn_7211_'(7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ<
Њ " џџџџџџџџџ*
&__inference_dense_1_layer_call_fn_7259_-.7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ*
Њ " џџџџџџџџџ
Џ
'__inference_conv2d_1_layer_call_fn_6954IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЕ
"__inference_signature_wrapper_7168
!"'(-.CЂ@
Ђ 
9Њ6
4
input_1)&
input_1џџџџџџџџџ  ";Њ8
6
output_1*'
output_1џџџџџџџџџ
