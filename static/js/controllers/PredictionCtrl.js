var app = angular.module("PredictionCtrl", []);

app.controller("PredictionController", function ($scope, $rootScope, $log,$http) {
	
	var company = $rootScope.inputCompany;
	
	$scope.predictstockprices = function(){
		
		$log.info('company ' + JSON.stringify(company));
		$http.post('/prediction',JSON.stringify(company)).then(function (response) {
			
			$log.info(response.data[0]);
			$log.info(response.data[1]);
			$scope.stockValue = response.data[0];
			$scope.stockValuePrev = response.data[1];
			$scope.stockValueActual = response.data[2];
			$scope.stringValue1 = 'The Stock Price of Symbol ' + $rootScope.inputCompany + ' was $' + $scope.stockValuePrev + ' day-before-yesterday.';
			$scope.stringValue2 = 'Our model predicts the price to be $' + $scope.stockValue + ' yesterday.';
			$scope.stringValue3 = 'The actual price yesterday was $' + $scope.stockValueActual + '.';
	})
	}
});
