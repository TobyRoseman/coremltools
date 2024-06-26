#include "MLModelTests.hpp"
#include "../src/Model.hpp"
#include "../src/Format.hpp"

#include "framework/TestUtils.hpp"

using namespace CoreML;

int testSpecDowngradePipeline() {

    int32_t latestVersion = MLMODEL_SPECIFICATION_VERSION;

    Specification::Model spec;
    spec.set_specificationversion(latestVersion);
    auto* pipeline = spec.mutable_pipelineclassifier()->mutable_pipeline();

    auto* input = spec.mutable_description()->add_input();
    input->set_name("image");
    input->mutable_type()->mutable_imagetype()->set_width(299);
    input->mutable_type()->mutable_imagetype()->set_height(299);
    input->mutable_type()->mutable_imagetype()->set_colorspace(Specification::ImageFeatureType_ColorSpace_BGR);

    auto* output = spec.mutable_description()->add_output();
    output->set_name("classLabel");
    output->mutable_type()->mutable_stringtype();

    spec.mutable_description()->set_predictedfeaturename("classLabel");

    // VisionFeaturePrint
    auto *featureModel = pipeline->add_models();
    featureModel->set_specificationversion(latestVersion);

    auto* featureInput = featureModel->mutable_description()->add_input();
    featureInput->set_name("image");
    featureInput->mutable_type()->mutable_imagetype()->set_width(299);
    featureInput->mutable_type()->mutable_imagetype()->set_height(299);
    featureInput->mutable_type()->mutable_imagetype()->set_colorspace(Specification::ImageFeatureType_ColorSpace_BGR);

    auto featureOutput = featureModel->mutable_description()->add_output();
    featureOutput->set_name("features");
    featureOutput->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType::FLOAT32);
    featureOutput->mutable_type()->mutable_multiarraytype()->add_shape(2048);


    featureModel->mutable_visionfeatureprint()->mutable_scene()->set_version(Specification::CoreMLModels::VisionFeaturePrint_Scene_SceneVersion_SCENE_VERSION_1);

    // Logistic regression
    auto *classifierModel = pipeline->add_models();
    classifierModel->set_specificationversion(latestVersion);

    auto classifierInput = classifierModel->mutable_description()->add_input();
    classifierInput->set_name("features");
    classifierInput->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType::FLOAT32);
    classifierInput->mutable_type()->mutable_multiarraytype()->add_shape(2048);

    auto classifierOutput = classifierModel->mutable_description()->add_output();
    classifierOutput->set_name("classLabel");
    classifierOutput->mutable_type()->mutable_stringtype();
    classifierModel->mutable_description()->set_predictedfeaturename("classLabel");

    auto glm = classifierModel->mutable_glmclassifier();
    glm->set_postevaluationtransform(::CoreML::Specification::GLMClassifier_PostEvaluationTransform_Logit);
    glm->add_offset(0.0);
    auto weights = glm->add_weights();
    for (int i=0; i<2048; i++) { weights->add_value(0.0); }
    glm->mutable_stringclasslabels()->add_vector("cat");
    glm->mutable_stringclasslabels()->add_vector("dog");

    // Constructing an CoreML::Model should downgrade spec on load
    Model model(spec);

    // Top level should be IOS 12 because it contains vision feature print
    ML_ASSERT_EQ(model.getProto().specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS12);

    // First model in pipeline is vision feature print and should have IOS12 spec version
    ML_ASSERT_EQ(model.getProto().pipelineclassifier().pipeline().models(0).specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS12);

    // Second model is just a GLM which has support in IOS 11
    ML_ASSERT_EQ(model.getProto().pipelineclassifier().pipeline().models(1).specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS11);

    return 0;
}

int testWordTaggerTransferLearningSpecIOS14() {
    Specification::Model spec;

    //initialization
    spec.set_specificationversion(MLMODEL_SPECIFICATION_VERSION);
    Specification::ModelDescription* interface = spec.mutable_description();
    Specification::Metadata* metadata = interface->mutable_metadata();
    metadata->set_shortdescription(std::string("This is a Word tagger model"));

    auto *input = interface->add_input();
    input->mutable_type()->mutable_stringtype();
    input->set_name(std::string("text"));

    auto *output1 = interface->add_output();
    output1->mutable_type()->mutable_sequencetype()->mutable_stringtype();
    output1->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_lowerbound(0);
    output1->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_upperbound(10000);
    output1->set_name(std::string("tags"));
    auto *output2 = interface->add_output();
    output2->mutable_type()->mutable_sequencetype()->mutable_int64type();
    output2->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_lowerbound(0);
    output2->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_upperbound(10000);
    output2->set_name(std::string("locations"));
    auto *output3 = interface->add_output();
    output3->mutable_type()->mutable_sequencetype()->mutable_int64type();
    output3->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_lowerbound(0);
    output3->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_upperbound(10000);
    output3->set_name(std::string("lengths"));
    auto *output4 = interface->add_output();
    output4->mutable_type()->mutable_sequencetype()->mutable_stringtype();
    output4->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_lowerbound(0);
    output4->mutable_type()->mutable_sequencetype()->mutable_sizerange()->set_upperbound(10000);
    output4->set_name(std::string("tokens"));

    auto *wordTagger = spec.mutable_wordtagger();
    wordTagger->set_language(std::string("en-US"));
    wordTagger->set_tokensoutputfeaturename(std::string("tokens"));
    wordTagger->set_tokentagsoutputfeaturename(std::string("tags"));
    wordTagger->set_tokenlocationsoutputfeaturename(std::string("locations"));
    wordTagger->set_tokenlengthsoutputfeaturename(std::string("lengths"));
    auto *tags = wordTagger->mutable_stringtags();
    tags->add_vector("PER");
    tags->add_vector("LOC");
    std::string modelData = "This is a dummy model";
    wordTagger->set_modelparameterdata(modelData);
    wordTagger->set_revision(3); //transfer learning using revision 3

    // Constructing an CoreML::Model should downgrade spec on load
    Model model1(spec);
    ML_ASSERT_EQ(model1.getProto().specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS14);

    wordTagger->set_revision(1);
    Model model2(spec);
    ML_ASSERT_EQ(model2.getProto().specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS12);

    return 0;
}

int testEmptyInputModel_downgradeToIOS18() {
    // This model doesn't have input. Such empty input models are supported in iOS18 and later.
    auto spec = Specification::Model();
    spec.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_NEWEST);
    auto* modelDescription = spec.mutable_description();

    auto output = modelDescription->add_output();
    output->set_name("output");
    output->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32);
    output->mutable_type()->mutable_multiarraytype()->add_shape(1);

    auto* net = spec.mutable_neuralnetwork();
    auto* layer = net->add_layers();
    layer->set_name("load_constantND_layer");
    layer->add_output("output");
    layer->mutable_loadconstantnd()->add_shape(1);
    layer->mutable_loadconstantnd()->mutable_data()->add_floatvalue(0.1f);

    // The model uses empty input, which was introduced in iOS18.
    Model emptyInputModel(spec);
    ML_ASSERT_EQ(emptyInputModel.getProto().specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS18);

    // Now, add an input.
    auto input = modelDescription->add_input();
    input->set_name("input");
    input->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32);
    input->mutable_type()->mutable_multiarraytype()->add_shape(1);

    // The model uses EXACT_ARRAY_MAPPING, which was introduced in iOS13. Other than that,
    // there is nothing special. We expect the downgrade utility sets it to iOS13.
    Model modelWithInput(spec);
    ML_ASSERT_EQ(modelWithInput.getProto().specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS13);

    return 0;
}

int testMultiFunctionModel_downgradeToIOS18() {
    // This model doesn't have input. Such empty input models are supported in iOS18 and later.
    auto spec = Specification::Model();
    spec.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_NEWEST);
    auto* modelDescription = spec.mutable_description();

    auto *function = modelDescription->add_functions();
    function->set_name("f");

    modelDescription->set_defaultfunctionname("f");

    auto input = function->add_input();
    input->set_name("input");
    input->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32);
    input->mutable_type()->mutable_multiarraytype()->add_shape(1);

    auto output = function->add_output();
    output->set_name("output");
    output->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32);
    output->mutable_type()->mutable_multiarraytype()->add_shape(1);

    spec.mutable_mlprogram();

    // The model uses multi-function description syntax, which was introduced in iOS18.
    Model multiFunctionModel(spec);
    ML_ASSERT_EQ(multiFunctionModel.getProto().specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS18);

    // Let's remove multi-function syntax and use the good old syntax.
    modelDescription->clear_functions();
    modelDescription->clear_defaultfunctionname();

    input = modelDescription->add_input();
    input->set_name("input");
    input->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32);
    input->mutable_type()->mutable_multiarraytype()->add_shape(1);

    output = modelDescription->add_output();
    output->set_name("output");
    output->mutable_type()->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32);
    output->mutable_type()->mutable_multiarraytype()->add_shape(1);

    // Now, the model is nothing special ML Program, which was introduced in iOS15.
    Model notMultiFunctionModel(spec);
    ML_ASSERT_EQ(notMultiFunctionModel.getProto().specificationversion(), MLMODEL_SPECIFICATION_VERSION_IOS15);

    return 0;
}
