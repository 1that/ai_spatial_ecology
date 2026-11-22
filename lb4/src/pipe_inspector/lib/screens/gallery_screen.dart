import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import 'package:path_provider/path_provider.dart';
import '../services/database_service.dart';
import '../models/inspection.dart';
import 'package:url_launcher/url_launcher.dart';

class GalleryScreen extends StatefulWidget {
  const GalleryScreen({super.key});

  @override
  _GalleryScreenState createState() => _GalleryScreenState();
}

class _GalleryScreenState extends State<GalleryScreen> {
  final DatabaseService _db = DatabaseService();
  List<InspectionRecord> _records = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadRecords();
  }

  Future<void> _loadRecords() async {
    final records = await _db.getAllInspections();
    setState(() {
      _records = records;
      _isLoading = false;
    });
  }

  Future<void> _exportGeoJson() async {
    // Создаем FeatureCollection
    final features = _records.map((r) => r.toGeoJson()).toList();

    final geojson = {
      'type': 'FeatureCollection',
      'name': 'pipe_inspections',
      'features': features,
      'metadata': {
        'generated': DateTime.now().toIso8601String(),
        'count': _records.length,
        'source': 'DrainEye Mobile App',
      },
    };

    // Сохраняем в файл
    final directory = await getApplicationDocumentsDirectory();
    final file = File('${directory.path}/inspections.geojson');
    await file.writeAsString(jsonEncode(geojson));

    // Шерим файл
    await Share.shareXFiles(
      [XFile(file.path)],
      text: 'Результаты инспекции дренажных систем',
    );
  }

  Future<void> _openInMap(double lat, double lng) async {
    if (lat == 0 && lng == 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('У записи нет координат')),
      );
      return;
    }

    final Uri geoUri = Uri.parse('geo:$lat,$lng?q=$lat,$lng');
    final Uri webUri = Uri.parse('https://www.google.com/maps/search/?api=1&query=$lat,$lng');

    try {
      bool opened = false;
      try {
        opened = await launchUrl(geoUri, mode: LaunchMode.externalApplication);
      } catch (_) {
      }
      if (!opened) {
        await launchUrl(webUri, mode: LaunchMode.externalApplication);
      }
    } catch (e) {
      debugPrint('Ошибка открытия карты: $e');
      // Выведем ошибку на экран, чтобы понять, в чем дело
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Не удалось открыть карты: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Галерея инспекций'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: _records.isNotEmpty ? _exportGeoJson : null,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _records.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.photo_library, size: 80, color: Colors.grey),
                      const SizedBox(height: 20),
                      const Text('Нет записей', style: TextStyle(fontSize: 18)),
                      const SizedBox(height: 10),
                      ElevatedButton(
                        onPressed: () => Navigator.pop(context),
                        child: const Text('Сделать первую инспекцию'),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  itemCount: _records.length,
                  itemBuilder: (ctx, index) {
                    final record = _records[index];
                    return Card(
                      margin: const EdgeInsets.all(8),
                      child: ListTile(
                        leading: CircleAvatar(
                          backgroundImage: FileImage(File(record.photoPath)),
                        ),
                        title: Text('Материал: ${record.material ?? "N/A"}'),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Дата: ${record.timestamp.toLocal()}'),
                            Text('Координаты: ${record.latitude.toStringAsFixed(4)}, '
                                '${record.longitude.toStringAsFixed(4)}'),
                            if (record.confidence != null)
                              Text('Уверенность: ${(record.confidence! * 100).toStringAsFixed(1)}%'),
                          ],
                        ),
                        trailing: Row( // Row для 2 кнопок
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                              icon: const Icon(Icons.map, color: Colors.blue),
                              onPressed: () => _openInMap(record.latitude, record.longitude),
                            ),
                            IconButton(
                              icon: const Icon(Icons.delete, color: Colors.red),
                              onPressed: () async {
                                await _db.deleteInspection(record.id);
                                _loadRecords();
                              },
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
    );
  }
}