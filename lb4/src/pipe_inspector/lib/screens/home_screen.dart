import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/classifier_service.dart';
import '../services/database_service.dart';
import '../models/inspection.dart';

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const HomeScreen({super.key, required this.cameras});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  final ClassifierService _classifier = ClassifierService();
  final DatabaseService _db = DatabaseService();

  Position? _currentPosition;
  StreamSubscription<Position>? _positionSubscription;
  String _gpsStatus = 'поиск...';
  bool _isLoading = true;
  bool _isClassifying = false;
  String _result = '';
  double _confidence = 0.0;

  @override
  void initState() {
    super.initState();
    // Начинаем следить за состоянием
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
    _initLocation();
    _loadModel();
  }

  Future<void> _initCamera() async {
    _controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller.initialize();
    setState(() {});
  }

  Future<void> _initLocation() async {
    await _checkPermissions();
    await _startLocationUpdates();
  }

  static const LocationSettings _locationSettings = LocationSettings(
    accuracy: LocationAccuracy.high,
    distanceFilter: 0,
  );

  void _setGpsStatus(String status) {
    if (mounted) setState(() => _gpsStatus = status);
  }

  void _updatePosition(Position position) {
    if (mounted) setState(() => _currentPosition = position);
  }

  Future<void> _startLocationUpdates() async {
    if (!await Geolocator.isLocationServiceEnabled()) {
      _setGpsStatus('геолокация выключена');
      return;
    }

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      _setGpsStatus('нет разрешения');
      return;
    }

    final lastKnown = await Geolocator.getLastKnownPosition();
    if (lastKnown != null) _updatePosition(lastKnown);

    await _positionSubscription?.cancel();
    _positionSubscription = Geolocator.getPositionStream(locationSettings: _locationSettings).listen(
      _updatePosition,
      onError: (e) {
        print('Ошибка потока координат: $e');
        _setGpsStatus('ошибка');
      },
    );
  }

  Future<void> _checkPermissions() async {
    await [
      Permission.camera,
      Permission.location,
      Permission.storage,
    ].request();
  }

  Future<void> _getCurrentLocation() async {
    try {
      final position = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 10), // чтобы приложение не зависло в ожидании
        ),
      );
      _updatePosition(position);
    } catch (e) {
      print('Ошибка получения координат: $e');
    }
  }

  Future<void> _loadModel() async {
    await _classifier.loadModel();
    setState(() {
      _isLoading = false;
    });
  }

  Future<void> _takePicture() async {
    if (_isClassifying) return;

    setState(() {
      _isClassifying = true;
      _result = '';
    });

    try {
      await _initializeControllerFuture;

      // Съемка
      final image = await _controller.takePicture();

      // Классификация
      final result = await _classifier.classifyImage(File(image.path));

      if (_currentPosition == null) {
        await _getCurrentLocation();
      }

      // Генерация ID
      final recordId = '${DateTime.now().millisecondsSinceEpoch}';

      // Сохранение записи
      final record = InspectionRecord(
        id: recordId,
        latitude: _currentPosition?.latitude ?? 0,
        longitude: _currentPosition?.longitude ?? 0,
        timestamp: DateTime.now(),
        photoPath: image.path,
        material: result['material'],
        confidence: result['confidence'],
        damagePercent: Random().nextDouble() * 50, // упрощенно
        synced: false,
      );

      await _db.insertInspection(record);

      setState(() {
        _result = result['material'];
        _confidence = result['confidence'];
      });

      // Показываем диалог с результатом
      _showResultDialog(result);
    } catch (e) {
      print('Ошибка: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Ошибка: $e')),
      );
    } finally {
      setState(() {
        _isClassifying = false;
      });
    }
  }

  void _showResultDialog(Map<String, dynamic> result) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Результат'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Материал: ${result['material']}'),
            Text('Уверенность: ${(result['confidence'] * 100).toStringAsFixed(1)}%'),
            if (_currentPosition != null)
              Text('Координаты: ${_currentPosition!.latitude.toStringAsFixed(4)}, '
                  '${_currentPosition!.longitude.toStringAsFixed(4)}'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('OK'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              _navigateToGallery();
            },
            child: const Text('Галерея'),
          ),
        ],
      ),
    );
  }

  void _navigateToGallery() {
    Navigator.pushNamed(context, '/gallery');
  }

  @override
  Widget build(BuildContext context) {
    // Проверка: если список камер пуст
    if (widget.cameras.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('DrainEye Inspector')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.videocam_off, size: 64, color: Colors.grey),
              const SizedBox(height: 16),
              const Text(
                'Камера не обнаружена',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              const Text('Проверьте разрешения или устройство'),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _navigateToGallery,
                child: const Text('Перейти в галерею'),
              ),
            ],
          ),
        ),
      );
    }

    // Если камеры есть, продолжаем обычную отрисовку
    return Scaffold(
      appBar: AppBar(
        title: const Text('DrainEye Inspector'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            onPressed: _navigateToGallery,
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {},
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return Stack(
                    children: [
                      CameraPreview(_controller),
                      Positioned(
                        top: 20,
                        left: 20,
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _currentPosition == null
                                    ? 'GPS: $_gpsStatus'
                                    : 'GPS: ${_currentPosition!.latitude.toStringAsFixed(4)}°, '
                                        '${_currentPosition!.longitude.toStringAsFixed(4)}°',
                                style: const TextStyle(color: Colors.white),
                              ),
                              if (_classifier.isLoaded)
                                const Text(
                                  'Модель: активна',
                                  style: TextStyle(color: Colors.green, fontSize: 12),
                                ),
                            ],
                          ),
                        ),
                      ),
                      if (_result.isNotEmpty)
                        Positioned(
                          bottom: 100,
                          left: 20,
                          right: 20,
                          child: Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Colors.black87,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Материал: $_result',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                LinearProgressIndicator(
                                  value: _confidence,
                                  backgroundColor: Colors.grey,
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    _confidence > 0.8
                                        ? Colors.green
                                        : _confidence > 0.6
                                            ? Colors.orange
                                            : Colors.red,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'Уверенность: ${(_confidence * 100).toStringAsFixed(1)}%',
                                  style: const TextStyle(color: Colors.white70),
                                ),
                              ],
                            ),
                          ),
                        ),
                      if (_isClassifying)
                        Positioned.fill(
                          child: Container(
                            color: Colors.black54,
                            child: const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  CircularProgressIndicator(),
                                  SizedBox(height: 20),
                                  Text(
                                    'Анализ изображения...',
                                    style: TextStyle(color: Colors.white),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                    ],
                  );
                } else {
                  return const Center(child: CircularProgressIndicator());
                }
              },
            ),
      floatingActionButton: FloatingActionButton(
        onPressed: _takePicture,
        child: const Icon(Icons.camera_alt),
      ),
    );
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this); // Перестаем следить
    _positionSubscription?.cancel();
    _controller.dispose(); // освобождаем камеру
    _classifier.dispose();
    super.dispose();
  }

  // метод, который реагирует на сворачивание приложения
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Если приложение свернуто, выключаем камеру, чтобы не было ошибки "dead thread"
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      // Если вернулись, инициализируем заново
      _initCamera();
    }
  }
}