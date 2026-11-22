import 'dart:io';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;

class ClassifierService {
  static const String _modelPath = 'assets/models/model_fp16.tflite';
  static const String _labelsPath = 'assets/models/labels.txt';

  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;

  Future<void> loadModel() async {
    try {
      // Загрузка модели
      _interpreter = await Interpreter.fromAsset(_modelPath);

      // Загрузка меток
      final labelsData = await rootBundle.loadString(_labelsPath);
      _labels = labelsData.split('\n').where((s) => s.trim().isNotEmpty).toList();

      _isLoaded = true;
      print('Модель загружена. Классов: ${_labels.length}');

      // Информация о модели
      var inputShape = _interpreter!.getInputTensor(0).shape;
      var outputShape = _interpreter!.getOutputTensor(0).shape;
      print('  Вход: $inputShape');
      print('  Выход: $outputShape');
    } catch (e) {
      print('Ошибка загрузки модели: $e');
    }
  }

  bool get isLoaded => _isLoaded;
  List<String> get labels => _labels;

  Future<Map<String, dynamic>> classifyImage(File imageFile) async {
    if (!_isLoaded) {
      throw Exception('Модель не загружена');
    }

    // Чтение и ресайз изображения
    final imageBytes = await imageFile.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);
    if (image == null) {
      throw Exception('Не удалось прочитать изображение');
    }

    // Ресайз до 224x224
    img.Image resized = img.copyResize(image, width: 224, height: 224);

    // Подготовка входного тензора
    var input = List.generate(
      1,
      (b) => List.generate(
        224,
        (h) => List.generate(
          224,
          (w) => List.generate(3, (c) => 0.0),
        ),
      ),
    );

    // Нормализация
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        var pixel = resized.getPixel(x, y);
        input[0][y][x][0] = pixel.r / 255.0;
        input[0][y][x][1] = pixel.g / 255.0;
        input[0][y][x][2] = pixel.b / 255.0;
      }
    }

    // Инференс
    var output = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);
    _interpreter!.run(input, output);

    // Поиск лучшего класса
    int maxIdx = 0;
    double maxVal = 0.0;
    for (int i = 0; i < _labels.length; i++) {
      if (output[0][i] > maxVal) {
        maxVal = output[0][i];
        maxIdx = i;
      }
    }

    return {
      'material': _labels[maxIdx],
      'confidence': maxVal,
    };
  }

  void dispose() {
    _interpreter?.close();
  }
}